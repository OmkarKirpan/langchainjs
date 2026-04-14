/* oxlint-disable @typescript-eslint/no-explicit-any */

/**
 * Agent-level streaming support (experimental).
 *
 * Provides AgentRunStream — a wrapper over GraphRunStream that adds
 * native projections for tool calls and middleware events — and the
 * transformer factories that power them.
 *
 * See protocol proposal §15 (In-Process Streaming Interface) and §16
 * (Native Stream Transformers) for the design behind this module.
 */

import {
  GraphRunStream,
  StreamChannel,
  type StreamTransformer,
  type ProtocolEvent,
  type ToolCallStream,
  type ToolCallStatus,
  type ToolsEventData,
  type UpdatesEventData,
  type ChatModelStream,
  type Namespace,
  type InterruptPayload,
} from "@langchain/langgraph";
import type {
  ClientTool,
  ServerTool,
  DynamicStructuredTool,
  StructuredToolInterface,
} from "@langchain/core/tools";

// ─── Tool type helpers ────────────────────────────────────────────────────────

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

// ─── MiddlewareEvent ──────────────────────────────────────────────────────────

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
 * Run stream for agent-level abstractions.
 *
 * Wraps a {@link GraphRunStream} from the underlying graph's `streamV2()`
 * and lifts tool call and middleware projections — registered as
 * extension transformers — into native getters.
 *
 * @typeParam TValues - Shape of the agent's merged state.
 * @typeParam TTools - Tuple of tool types; enables typed
 *   {@link ToolCallStream} narrowing on `run.toolCalls`.
 */
export class AgentRunStream<
  TValues = Record<string, unknown>,
  TTools extends readonly (ClientTool | ServerTool)[] = readonly (
    | ClientTool
    | ServerTool
  )[],
> implements AsyncIterable<ProtocolEvent> {
  readonly #inner: GraphRunStream<TValues, any>;
  readonly #toolCallsIterable: AsyncIterable<ToolCallStreamUnion<TTools>>;
  readonly #middlewareIterable: AsyncIterable<MiddlewareEvent>;

  constructor(
    inner: GraphRunStream<TValues, any>,
    toolCallsIterable: AsyncIterable<ToolCallStreamUnion<TTools>>,
    middlewareIterable: AsyncIterable<MiddlewareEvent>
  ) {
    this.#inner = inner;
    this.#toolCallsIterable = toolCallsIterable;
    this.#middlewareIterable = middlewareIterable;
  }

  get path(): Namespace {
    return this.#inner.path;
  }

  get extensions(): Record<string, unknown> {
    return this.#inner.extensions;
  }

  [Symbol.asyncIterator](): AsyncIterator<ProtocolEvent> {
    return this.#inner[Symbol.asyncIterator]();
  }

  get subgraphs(): AsyncIterable<
    import("@langchain/langgraph").SubgraphRunStream
  > {
    return this.#inner.subgraphs;
  }

  get values(): AsyncIterable<TValues> & PromiseLike<TValues> {
    return this.#inner.values;
  }

  get messages(): AsyncIterable<ChatModelStream> {
    return this.#inner.messages;
  }

  messagesFrom(node: string): AsyncIterable<ChatModelStream> {
    return this.#inner.messagesFrom(node);
  }

  get output(): Promise<TValues> {
    return this.#inner.output;
  }

  get interrupted(): boolean {
    return this.#inner.interrupted;
  }

  get interrupts(): readonly InterruptPayload[] {
    return this.#inner.interrupts;
  }

  abort(reason?: unknown): void {
    this.#inner.abort(reason);
  }

  get signal(): AbortSignal {
    return this.#inner.signal;
  }

  /**
   * Yields one {@link ToolCallStream} per tool invocation observed in
   * the run. Tool calls are emitted when the model finishes generating
   * arguments (before execution begins).
   *
   * When the agent's tools are typed (e.g. via `createAgent({ tools })`
   * with literal inference), narrowing by `call.name` gives typed
   * `.input` and `.output`.
   */
  get toolCalls(): AsyncIterable<ToolCallStreamUnion<TTools>> {
    return this.#toolCallsIterable;
  }

  /**
   * Yields one {@link MiddlewareEvent} per middleware lifecycle
   * transition (before/after agent and model nodes).
   */
  get middleware(): AsyncIterable<MiddlewareEvent> {
    return this.#middlewareIterable;
  }
}

interface ToolCallProjection {
  _toolCalls: StreamChannel<ToolCallStream>;
}

function hasPrefix(ns: Namespace, prefix: Namespace): boolean {
  if (prefix.length > ns.length) return false;
  for (let i = 0; i < prefix.length; i += 1) {
    if (ns[i] !== prefix[i]) return false;
  }
  return true;
}

/**
 * Creates a transformer that correlates `tools` channel events into
 * per-call {@link ToolCallStream} objects.
 *
 * Uses {@link StreamChannel} so that tool call data is available both
 * in-process (via `run.toolCalls`) and to remote clients (auto-forwarded
 * as protocol events on the `"toolCalls"` channel). The mux handles
 * close/fail of the channel automatically; `finalize`/`fail` only
 * clean up pending promises on in-flight tool calls.
 *
 * A tool call is created on `tool-started` (name and input available).
 * The `.output`, `.status`, and `.error` promises are resolved when
 * `tool-finished` or `tool-error` arrives for the same `tool_call_id`.
 *
 * For models that emit `content-block-finish` with `type: "tool_call"`
 * on the `messages` channel (e.g. Anthropic streaming), tool calls are
 * also captured from there — whichever event arrives first creates the
 * stream; duplicates are ignored via the pending calls map.
 */
export function createToolCallTransformer(
  path: Namespace
): () => StreamTransformer<ToolCallProjection> {
  return () => {
    const toolCallsCh = new StreamChannel<ToolCallStream>("toolCalls");

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

      toolCallsCh.push({
        name,
        callId,
        input,
        output,
        status,
        error,
      } as ToolCallStream);
    }

    return {
      init: () => ({
        _toolCalls: toolCallsCh,
      }),

      process(event: ProtocolEvent): boolean {
        if (!hasPrefix(event.params.namespace, path)) return true;

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
  _middleware: StreamChannel<MiddlewareEvent>;
}

const MIDDLEWARE_NODE_PATTERN =
  /^(.+)\.(before_agent|before_model|after_model|after_agent)$/;

/**
 * Creates a transformer that watches `updates` events from
 * middleware nodes and surfaces them as typed {@link MiddlewareEvent}
 * objects.
 *
 * Uses {@link StreamChannel} so that middleware events are available both
 * in-process (via `run.middleware`) and to remote clients (auto-forwarded
 * as protocol events on the `"middleware"` channel). The mux handles
 * close/fail of the channel automatically — no `finalize`/`fail` needed.
 *
 * Middleware nodes follow the naming convention
 * `<middleware_name>.<phase>` (e.g. `summarization.before_model`).
 */
export function createMiddlewareTransformer(
  path: Namespace
): () => StreamTransformer<MiddlewareProjection> {
  return () => {
    const middleware = new StreamChannel<MiddlewareEvent>("middleware");

    return {
      init: () => ({
        _middleware: middleware,
      }),

      process(event: ProtocolEvent): boolean {
        if (event.method !== "updates") return true;
        if (!hasPrefix(event.params.namespace, path)) return true;

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
