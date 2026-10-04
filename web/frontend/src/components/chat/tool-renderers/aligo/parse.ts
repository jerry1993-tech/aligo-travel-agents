import type { ToolResultBlock } from '@agentscope-ai/agentscope/message';

import { getResultText } from '../_shared';

/**
 * 差旅工具返回值的**卡片类型常量**。
 *
 * ⚠️ 这些值必须与后端 ``src/tools/_result.py`` 里的 ``CARD_*`` 常量**逐字**
 * 一致。两边对不上时的表现是**静默的**：后端照常返回 ``card`` 字段，
 * 前端查不到对应渲染器就退回默认渲染（一段 JSON 文本），不报错、不告警，
 * 只是卡片没了。改这里时务必同步改后端常量（或反过来）。
 */
export const CARD_ROUTE = 'route_decision';
export const CARD_TRANSPORT = 'transport_options';
export const CARD_HOTEL = 'hotel_options';
export const CARD_POLICY = 'policy_verdict';
export const CARD_ORDERS = 'order_list';
export const CARD_APPROVAL = 'approval_result';

/**
 * 差旅工具统一返回载荷的形状（见 ``src/tools/_result.py`` 的模块文档）::
 *
 *     { ok, summary, card, items, needs?, detail?, ...其余顶层字段 }
 *
 * 这里刻意**不穷举**各卡片特有的顶层字段（如 ``search_transport`` 的
 * ``origin``/``destination``）：它们是按工具补的，用索引签名让调用方按需读取，
 * 好过在这里维护一张随工具增删而漂移的字段表。
 */
export interface AligoResult {
	ok: boolean;
	/** 面向人/模型的一句话中文摘要。 */
	summary: string;
	/** 卡片类型；空串表示「不需要卡片」。 */
	card: string;
	/** 卡片数据；``null`` 表示该工具不产出列表（区别于产出空列表 ``[]``）。 */
	items: unknown[] | null;
	/** 仅「信息不足需追问」时出现：缺少的要素名。 */
	needs?: string[];
	/**
	 * 仅失败时出现：**面向排障**的补充信息（内部标识与异常原文）。
	 *
	 * ⚠️ 保留在类型里是为了说明载荷形状，但**任何渲染器都不要读它** ——
	 * 它不是给用户看的内容。2026-10-03 的审计发现 `shell.tsx` 曾把它
	 * 用等宽字体渲染给用户，已删除；服务端改为写日志。
	 */
	detail?: string;
	[key: string]: unknown;
}

/**
 * 从工具结果里解析出差旅载荷。
 *
 * ⚠️ 用 ``try/catch`` 兜底而不是假设一定成功。三种情况都会走到这里：
 *   1. 结果还在流式流入（此时文本是半截 JSON）；
 *   2. 后端某个工具**没有**遵循 :mod:`src/tools/_result` 契约、返回了纯文本；
 *   3. 结果根本不是我们的工具产的（比如 MCP 工具），只是恰好命中了同一个
 *      工具名分发。
 * 无论哪种，返回 ``null`` 让调用方退回默认渲染即可 —— **绝不能抛异常**：
 * 这是渲染路径，抛出去会让整条消息（连同其它已渲染的工具行）一起崩掉。
 *
 * @param result - 工具结果块，可能为空（调用还没产生结果）。
 * @returns 解析成功的载荷；解析不出则 ``null``。
 */
export function parseAligoResult(result?: ToolResultBlock): AligoResult | null {
	if (!result) return null;
	const text = getResultText(result).trim();
	if (!text) return null;
	try {
		const parsed: unknown = JSON.parse(text);
		// 只认「对象」。数组或标量不可能是我们的契约载荷，直接判失败。
		if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
			return parsed as AligoResult;
		}
		return null;
	} catch {
		return null;
	}
}

/** 把一个未知值当对象读，非对象则给出空对象，省去调用处反复判断。 */
export function asRecord(value: unknown): Record<string, unknown> {
	return value && typeof value === 'object' && !Array.isArray(value)
		? (value as Record<string, unknown>)
		: {};
}

/** 把未知值读成字符串；缺失/非字符串时返回 `fallback`。 */
export function asString(value: unknown, fallback = ''): string {
	return typeof value === 'string' ? value : fallback;
}

/** 把未知值读成数字；缺失/非数字/NaN 时返回 `fallback`。 */
export function asNumber(value: unknown, fallback = 0): number {
	if (typeof value === 'number' && Number.isFinite(value)) return value;
	if (typeof value === 'string' && value.trim() !== '') {
		const n = Number(value);
		if (Number.isFinite(n)) return n;
	}
	return fallback;
}

/** 把未知值读成数组；非数组时返回空数组。 */
export function asArray(value: unknown): unknown[] {
	return Array.isArray(value) ? value : [];
}

/** 把 ``items`` 读成一串记录，跳过其中不是对象的元素。 */
export function itemRecords(data: AligoResult): Record<string, unknown>[] {
	return asArray(data.items).map(asRecord);
}

/**
 * 金额格式化：``1280`` → ``¥1280``。
 *
 * ⚠️ 用 ``Number.isInteger`` 决定是否保留小数，而不是无脑 ``toFixed(2)``：
 * 差旅单价基本都是整数元，``¥1280`` 比 ``¥1280.00`` 好读；真有小数
 * （比如按汇率换算的结果）也不至于被抹掉。
 */
export function formatMoney(value: unknown): string {
	const n = asNumber(value, 0);
	return `¥${Number.isInteger(n) ? n : n.toFixed(2)}`;
}

/**
 * 时长格式化：``155`` → ``2h35m``；不足一小时只给分钟。
 *
 * ⚠️ 不做本地化文案（不写「2 小时 35 分」）：这个串在窄卡片里要跟
 * 起降时间并排，短比"正确"重要；且它是**量级**提示，不是给人精算的。
 */
export function formatDuration(minutes: unknown): string {
	const total = Math.max(0, Math.round(asNumber(minutes, 0)));
	const h = Math.floor(total / 60);
	const m = total % 60;
	if (h > 0) return m > 0 ? `${h}h${m}m` : `${h}h`;
	return `${m}m`;
}
