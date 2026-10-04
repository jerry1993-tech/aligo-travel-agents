import type { ToolCallBlock } from '@agentscope-ai/agentscope/message';
import type { ReactNode } from 'react';

import { ToolCallRow } from './_shared';
import { BashRenderer } from './BashRenderer';
import {
	defaultGetDisplayName,
	defaultRenderBody,
	defaultRenderConfirmBody,
	defaultRenderHeader,
} from './DefaultRenderer';
import { EditRenderer } from './EditRenderer';
import { GlobRenderer } from './GlobRenderer';
import { GrepRenderer } from './GrepRenderer';
import { ReadRenderer } from './ReadRenderer';
import { TaskCreateRenderer } from './TaskCreateRenderer';
import type { TFunction, ToolCallWithResult, ToolRenderer } from './types';
import { WriteRenderer } from './WriteRenderer';
import { ALIGO_RENDERERS } from './aligo';

/**
 * 工具名 → 渲染器。
 *
 * ⚠️ 差旅工具的渲染器**成组**放在 ``./aligo``，这里用展开合并进来，而不是把
 * 7 个工具名逐个写死在这张表里。理由是「归属」：框架内建工具（Bash/Read/
 * ...）与本项目自有的差旅工具是两拨东西，混在一张平铺的表里，将来想整体
 * 移除或改写差旅那块时无从下手。展开合并在运行时与逐个写等价，但让「这一
 * 块是差旅的」在源码里可见。
 *
 * ⚠️ 展开放在**后面**：若哪天工具重名，后写的会覆盖先写的。当前两边工具名
 * 不相交（差旅工具都带业务前缀或是 search_/query_ 这类语义名），不会发生；
 * 万一将来撞名，让差旅的赢反而可能掩盖框架工具被吞掉的问题 —— 所以这条
 * 顺序值得保留意识：**新增框架工具时确认其名不在差旅工具清单里**。
 */
const renderers: Record<string, ToolRenderer> = {
	Bash: BashRenderer,
	Read: ReadRenderer,
	Write: WriteRenderer,
	Edit: EditRenderer,
	Glob: GlobRenderer,
	Grep: GrepRenderer,
	TaskCreate: TaskCreateRenderer,
	...ALIGO_RENDERERS,
};

function getRenderer(toolName: string): ToolRenderer {
	return renderers[toolName] ?? {};
}

export function getDisplayName(call: ToolCallBlock, t: TFunction): string {
	const r = getRenderer(call.name);
	return r.getDisplayName?.(call, t) ?? defaultGetDisplayName(call);
}

export function renderConfirmBody(call: ToolCallBlock, t: TFunction): ReactNode {
	const r = getRenderer(call.name);
	return r.renderConfirmBody?.(call, t) ?? defaultRenderConfirmBody(call);
}

/**
 * Render a single tool call as one collapsible row. The tool's renderer only
 * supplies the trigger-line `header` and the expandable `body`; the shared
 * `ToolCallRow` owns the collapsible shell, state icon and chevron. Falls back
 * to the `Default*` implementations for tools without a dedicated renderer.
 */
export function renderToolCall(pair: ToolCallWithResult, t: TFunction): ReactNode {
	const r = getRenderer(pair.call.name);
	const header = r.renderHeader?.(pair, t) ?? defaultRenderHeader(pair, t);
	const body = r.renderBody?.(pair, t) ?? defaultRenderBody(pair, t);
	return <ToolCallRow key={pair.call.id} pair={pair} header={header} body={body} />;
}
