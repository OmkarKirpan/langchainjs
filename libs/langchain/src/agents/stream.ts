/* oxlint-disable @typescript-eslint/no-explicit-any */

/**
 * Agent-level streaming support (experimental).
 *
 * Provides native stream transformer factories for tool calls and
 * middleware events.  When marked `__native: true`, their projections
 * are assigned directly onto the `GraphRunStream` instance by
 * `createGraphRunStream` in langgraph-core — no subclass or wrapper
 * needed.
 *
 * See protocol proposal §15 (In-Process Streaming Interface) and §16
 * (Native Stream Transformers).
 */

import {
  GraphRunStream,
  StreamChannel,
  type NativeStreamTransformer,
  type ProtocolEvent,
  type ToolCallStream,
  type ToolCallStatus,
  type ToolsEventData,
  type UpdatesEventData,
  type Namespace,
} from "@langchain/langgraph";
import type {
  ClientTool,
  ServerTool,
  DynamicStructuredTool,
  StructuredToolInterface,
} from "@langchain/core/tools";

/** Extract the literal `name` string from a tool type. */
type ToolNameOf<T> = T extends { name: infer N extends string } ? N : string;

/** Extract the parsed input type from a tool type. */
type ToolInputOf<T> =
  T extends DynamicStructuredTool<any, any, infer SchemaInputT, any, any, any>
    ? SchemaInputT
    : T extends StructuredToolInterface<any, infer SchemaInputT, any>
      ? SchemaInputT
      : unknown;

/** Extract the return/output type from a tool type. */
type ToolOutputOf<T> =
  T extends DynamicStructuredTool<any, any, any, infer ToolOutputT, any, any>
    ? ToolOutputT
    : T extends StructuredToolInterface<any, any, infer ToolOutputT>
      ? ToolOutputT
      : unknown;

/**
 * Discriminated union of {@link ToolCallStream} variants, one per tool
 * in `TTools`.  Enables TypeScript to narrow `.input` and `.output`
 * when the consumer checks `call.name === "someToolName"`.
 *
 * Falls back to `ToolCallStream` (untyped) when the tools tuple is a
 * plain `(ClientTool | ServerTool)[]` without literal name types.
 */
export type ToolCallStreamUnion<
  TTools extends readonly (ClientTool | ServerTool)[],
> = {
  [K in keyof TTools]: ToolCallStream<
    ToolNameOf<TTools[K]>,
    ToolInputOf<TTools[K]>,
    ToolOutputOf<TTools[K]>
  >;
}[number];

/**
 * Lifecycle phase that a middleware hook occupies within an agent turn.
 */
export type MiddlewarePhase =
  | "before_agent"
  | "before_model"
  | "after_model"
  | "after_agent";

/**
 * Represents a single middleware lifecycle event observed during an
 * agent run. Emitted by the middleware transformer.
 */
export interface MiddlewareEvent {
  phase: MiddlewarePhase;
  middlewareName: string;
  stateDelta: Record<string, unknown>;
  timestamp: number;
}

/**
 * A {@link GraphRunStream} with native agent-level projections assigned
 * directly on the instance by `createGraphRunStream` (via `__native`
 * transformers).
 *
 * This is a pure type overlay — no runtime subclass exists.  Use the
 * `AgentRunStream` type when you need to describe the return type of
 * `stream_experimental()`.
 */
export type AgentRunStream<
  TValues = Record<string, unknown>,
  TTools extends readonly (ClientTool | ServerTool)[] = readonly (
    | ClientTool
    | ServerTool
  )[],
> = GraphRunStream<TValues, any> & {
  /** Tool call streams from the native ToolCallTransformer. */
  toolCalls: AsyncIterable<ToolCallStreamUnion<TTools>>;
  /** Middleware lifecycle events from the native MiddlewareTransformer. */
  middleware: AsyncIterable<MiddlewareEvent>;
};

interface ToolCallProjection {
  toolCalls: StreamChannel<ToolCallStream>;
}

/**
 * Returns true when `ns` is at exactly the same depth as `path` (not a
 * child namespace). Used by agent-level transformers so that
 * `run.toolCalls` / `run.middleware` only contain events from the
 * agent's own graph, not from subagent subgraphs.
 */
function isAtDepth(ns: Namespace, path: Namespace): boolean {
  if (ns.length !== path.length) return false;
  for (let i = 0; i < path.length; i += 1) {
    if (ns[i] !== path[i]) return false;
  }
  return true;
}

/**
 * Creates a native transformer that correlates `tools` channel events
 * into per-call {@link ToolCallStream} objects.
 *
 * Marked `__native: true` — projection keys land directly on the
 * `GraphRunStream` instance as `run.toolCalls`.
 */
export function createToolCallTransformer(
  path: Namespace
): () => NativeStreamTransformer<ToolCallProjection> {
  return () => {
    const toolCalls = new StreamChannel<ToolCallStream>("toolCalls");

    const pendingCalls = new Map<
      string,
      {
        resolveOutput: (v: unknown) => void;
        rejectOutput: (e: unknown) => void;
        resolveStatus: (v: ToolCallStatus) => void;
        resolveError: (v: string | undefined) => void;
      }
    >();

    function createToolCallEntry(
      callId: string,
      name: string,
      input: unknown
    ): void {
      if (pendingCalls.has(callId)) return;

      let resolveOutput!: (v: unknown) => void;
      let rejectOutput!: (e: unknown) => void;
      let resolveStatus!: (v: ToolCallStatus) => void;
      let resolveError!: (v: string | undefined) => void;

      const output = new Promise<unknown>((res, rej) => {
        resolveOutput = res;
        rejectOutput = rej;
      });
      const status = new Promise<ToolCallStatus>((res) => {
        resolveStatus = res;
      });
      const error = new Promise<string | undefined>((res) => {
        resolveError = res;
      });

      pendingCalls.set(callId, {
        resolveOutput,
        rejectOutput,
        resolveStatus,
        resolveError,
      });

      toolCalls.push({
        name,
        callId,
        input,
        output,
        status,
        error,
      } as ToolCallStream);
    }

    return {
      __native: true as const,

      init: () => ({
        toolCalls,
      }),

      process(event: ProtocolEvent): boolean {
        /**
         * Only process events that are at the same depth as the agent's graph.
         */
        if (!isAtDepth(event.params.namespace, path)) return true;

        if (event.method === "messages") {
          const data = event.params.data as Record<string, unknown>;
          if (data.event === "content-block-finish") {
            const cb = (data.contentBlock ?? data.content_block) as
              | Record<string, unknown>
              | undefined;
            if (cb?.type === "tool_call") {
              createToolCallEntry(
                String(cb.id ?? ""),
                String(cb.name ?? ""),
                cb.args ?? cb.input
              );
            }
          }
        }

        if (event.method === "tools") {
          const data = event.params.data as ToolsEventData;
          const toolCallId = (data as Record<string, unknown>)
            .tool_call_id as string;

          if (data.event === "tool-started") {
            createToolCallEntry(
              toolCallId,
              ((data as Record<string, unknown>).tool_name as string) ??
                "unknown",
              (data as Record<string, unknown>).input
            );
          }

          const pending = toolCallId ? pendingCalls.get(toolCallId) : undefined;

          if (pending) {
            if (data.event === "tool-finished") {
              pending.resolveOutput((data as Record<string, unknown>).output);
              pending.resolveStatus("finished");
              pending.resolveError(undefined);
              pendingCalls.delete(toolCallId);
            } else if (data.event === "tool-error") {
              const message =
                ((data as Record<string, unknown>).message as string) ??
                "unknown error";
              pending.rejectOutput(new Error(message));
              pending.resolveStatus("error");
              pending.resolveError(message);
              pendingCalls.delete(toolCallId);
            }
          }
        }

        return true;
      },

      finalize(): void {
        for (const pending of pendingCalls.values()) {
          pending.resolveStatus("error");
          pending.resolveError("run finalized before tool completed");
          pending.rejectOutput(
            new Error("run finalized before tool completed")
          );
        }
        pendingCalls.clear();
      },

      fail(err: unknown): void {
        for (const pending of pendingCalls.values()) {
          pending.resolveStatus("error");
          pending.resolveError(
            err instanceof Error ? err.message : String(err)
          );
          pending.rejectOutput(err);
        }
        pendingCalls.clear();
      },
    };
  };
}

interface MiddlewareProjection {
  middleware: StreamChannel<MiddlewareEvent>;
}

const MIDDLEWARE_NODE_PATTERN =
  /^(.+)\.(before_agent|before_model|after_model|after_agent)$/;

/**
 * Creates a native transformer that watches `updates` events from
 * middleware nodes and surfaces them as typed {@link MiddlewareEvent}
 * objects.
 *
 * Marked `__native: true` — projection key lands directly on the
 * `GraphRunStream` instance as `run.middleware`.
 */
export function createMiddlewareTransformer(
  path: Namespace
): () => NativeStreamTransformer<MiddlewareProjection> {
  return () => {
    const middleware = new StreamChannel<MiddlewareEvent>("middleware");

    return {
      __native: true as const,

      init: () => ({
        middleware,
      }),

      process(event: ProtocolEvent): boolean {
        if (event.method !== "updates") return true;
        if (!isAtDepth(event.params.namespace, path)) return true;

        const data = event.params.data as UpdatesEventData;
        const nodeName = data.node ?? event.params.node;
        if (!nodeName) return true;

        const match = MIDDLEWARE_NODE_PATTERN.exec(nodeName);
        if (!match) return true;

        const middlewareName = match[1];
        const phase = match[2] as MiddlewarePhase;

        middleware.push({
          phase,
          middlewareName,
          stateDelta: data.values ?? {},
          timestamp: event.params.timestamp,
        });

        return true;
      },
    };
  };
}
