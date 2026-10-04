import type { ToolCallBlock } from '@agentscope-ai/agentscope/message';
import type { ComponentType, ReactNode } from 'react';

import { ApprovalResultCard } from './ApprovalResultCard';
import { HotelOptionsCard } from './HotelOptionsCard';
import { IntentDecisionsCard } from './IntentDecisionsCard';
import { OrderListCard } from './OrderListCard';
import type { AligoResult } from './parse';
import {
	parseAligoResult,
	CARD_APPROVAL,
	CARD_HOTEL,
	CARD_ORDERS,
	CARD_POLICY,
	CARD_ROUTE,
	CARD_TRANSPORT,
} from './parse';
import { PolicyVerdictCard } from './PolicyVerdictCard';
import { RouteDecisionCard } from './RouteDecisionCard';
import { NoticeBody } from './shell';
import { TransportOptionsCard } from './TransportOptionsCard';
import { defaultRenderBody, defaultRenderConfirmBody } from '../DefaultRenderer';
import { parseInput, toolArgClass, toolLabelClass } from '../_shared';
import type { TFunction, ToolRenderer } from '../types';

/**
 * ═══ 差旅卡片渲染器：按**工具名**分发，按**card 字段**决定画什么 ═══
 *
 * 这里存在一个真实的分歧，必须写清楚，否则后来人会以为其中一边写错了：
 *
 * - **后端契约**（``src/tools/_result.py`` 的 ``CARD_KEY`` 说明）要求
 *   「前端按结果 JSON 的 ``card`` **值**选卡片」，理由是同一个工具在不同
 *   阶段可能想给不同卡片，绑死在工具名上表达不了。
 * - **示例前端**（本目录的父 ``index.tsx``）是「按**工具名**分发」，
 *   与框架的 ``ToolRenderer`` 体系一致。
 *
 * 二者不一致的后果是**静默的**：卡片不会报错，只是悄悄退回默认渲染
 * （一段裸 JSON 文本）—— 排查时要从「卡片为什么没出来」一路倒推。
 *
 * 采用的方案是**两者都满足**：
 *   1. 仍按工具名注册到父 ``renderers`` 表 —— 改动最小，且完全沿用框架的
 *      ``ToolRenderer`` 抽象（``getDisplayName``/``renderHeader``/``renderBody``），
 *      不新造一套分发机制。
 *   2. 每个差旅工具的 ``renderBody`` **解析结果 JSON 的 ``card`` 字段**，
 *      再据此选卡片组件（见下面的 ``CARD_VIEWS``）——
 *      于是后端「按 card 值选卡片」的约定同样成立。
 *
 * 这样即便将来某个工具改了它返回的 ``card``（比如 ``search_transport``
 * 从 ``transport_options`` 改给别的卡片），后端改一个常量即可生效，
 * 前端无需改动；而框架层面的工具名分发保持不变。
 */

/** 卡片 ``card`` 值 → 正文组件。键必须与 ``parse.ts`` 的 ``CARD_*`` 常量一致。 */
const CARD_VIEWS: Record<string, ComponentType<{ data: AligoResult; t: TFunction }>> = {
	[CARD_ROUTE]: RouteDecisionCard,
	[CARD_TRANSPORT]: TransportOptionsCard,
	[CARD_HOTEL]: HotelOptionsCard,
	[CARD_POLICY]: PolicyVerdictCard,
	[CARD_ORDERS]: OrderListCard,
	[CARD_APPROVAL]: ApprovalResultCard,
};

/** 从工具入参里取一段用于触发行的展示文本；入参还在流式流入时返回空串。 */
type ArgGetter = (input: string) => string;

function field(input: string, key: string): string {
	const value = parseInput(input)[key];
	return typeof value === 'string' ? value : typeof value === 'number' ? String(value) : '';
}

/**
 * 差旅工具注册表：工具名 → 触发行标签的 i18n key 与入参摘要取法。
 *
 * ⚠️ 工具名必须与后端 ``src/tools/`` 里 ``FunctionTool`` 生成的注册名**逐字**
 * 一致（``FunctionTool`` 默认用函数名）。拼错一个字不会报错，只是这个工具
 * 永远走默认渲染 —— 与 :mod:`src/tools/route` 文档里「静默失效」是同一类坑。
 */
const ALIGO_TOOLS: { name: string; i18n: string; getArg: ArgGetter }[] = [
	{ name: 'aligo_route_intent', i18n: 'routeIntent', getArg: (i) => field(i, 'intent') },
	{ name: 'recognize_intent', i18n: 'recognizeIntent', getArg: (i) => field(i, 'text') },
	{
		name: 'search_transport',
		i18n: 'searchTransport',
		getArg: (i) => [field(i, 'origin'), field(i, 'destination')].filter(Boolean).join(' → '),
	},
	{ name: 'search_hotels', i18n: 'searchHotels', getArg: (i) => field(i, 'city') },
	{ name: 'check_travel_policy', i18n: 'checkTravelPolicy', getArg: (i) => field(i, 'kind') },
	{ name: 'query_orders', i18n: 'queryOrders', getArg: (i) => field(i, 'kind') },
	{ name: 'submit_approval', i18n: 'submitApproval', getArg: (i) => field(i, 'title') },
];

/**
 * ``submit_approval`` 的确认正文。
 *
 * ⚠️ 有副作用的工具（后端未标 ``is_read_only``）执行前会弹 ``ConfirmCard``，
 * 它调的是 ``renderConfirmBody``。默认实现只会把入参 JSON 原样铺出来 —— 对
 * 一张「要不要提交这张申请单」的确认卡来说太含糊了。这里把关键字段拆成
 * 中文标签的行，让用户在点「是」之前**看清自己即将提交什么**。
 */
function renderApprovalConfirmBody(call: ToolCallBlock, t: TFunction): ReactNode {
	const { title, destination, depart_date, days, amount } = parseInput(call.input) as Record<
		string,
		unknown
	>;
	const rows: [string, unknown][] = [
		[t('toolCard.approval.subject'), title],
		[t('toolCard.approval.destination'), destination],
		[t('toolCard.approval.depart'), depart_date],
		[t('toolCard.approval.days'), days],
		[t('toolCard.approval.amount'), amount],
	];
	const visible = rows.filter(([, value]) => value !== undefined && value !== '' && value !== 0);
	// 一条字段都还没流进来时退回默认正文（显示原始入参），别弹一张空卡片。
	if (visible.length === 0) return defaultRenderConfirmBody(call);
	return (
		<div className="flex flex-col gap-y-1">
			{visible.map(([label, value]) => (
				<div key={label} className="flex gap-x-2">
					<span className="text-muted-foreground shrink-0">{label}</span>
					<span className="break-all">{String(value)}</span>
				</div>
			))}
		</div>
	);
}

/**
 * 为单个差旅工具造一个 ``ToolRenderer``。
 *
 * 正文的判定顺序（顺序本身是契约的一部分）：
 *   1. 还没结果 / 运行中 / 等待确认 → 交回默认渲染（与其它工具状态表现一致）。
 *   2. 解析不出我们的载荷 → 交回默认渲染（见 ``parseAligoResult`` 的说明）。
 *   3. ``ok === false``（失败）或带 ``needs``（缺要素）→ 用 ``NoticeBody``，
 *      这两类结果的 ``card`` 都是空串，走不到卡片表。
 *   4. 有 ``card`` 且命中 ``CARD_VIEWS`` → 渲染对应卡片。
 *   5. ``recognize_intent`` 这种「无卡片但结果值得结构化展示」的 → 兜底视图。
 *   6. 都不命中 → 交回默认渲染。
 */
function makeAligoRenderer(tool: (typeof ALIGO_TOOLS)[number]): ToolRenderer {
	const { name, i18n, getArg } = tool;
	return {
		getDisplayName: (_call, t) => t(`tool.aligo.${i18n}`),

		// 只有 submit_approval 会用到确认卡；其余有副作用的工具目前不存在。
		renderConfirmBody: name === 'submit_approval' ? renderApprovalConfirmBody : undefined,

		renderHeader: (pair, t) => (
			<>
				<span className={toolLabelClass}>{t(`tool.aligo.${i18n}`)}</span>
				<span className={toolArgClass}>{getArg(pair.call.input)}</span>
			</>
		),

		renderBody: (pair, t) => {
			const { call, result } = pair;
			if (!result) return null;
			if (call.state === 'asking' || result.state === 'running') {
				return defaultRenderBody(pair, t);
			}

			const data = parseAligoResult(result);
			if (!data) return defaultRenderBody(pair, t);

			if (data.ok === false || (data.needs?.length ?? 0) > 0) {
				return <NoticeBody data={data} t={t} />;
			}

			// ``recognize_intent`` 没有 card，但它的 items 是结构化意图列表，
			// 值得比默认的 JSON 文本更好看的呈现 —— 兜底到专用视图。
			const View =
				CARD_VIEWS[data.card] ??
				(name === 'recognize_intent' ? IntentDecisionsCard : undefined);
			if (View) return <View data={data} t={t} />;

			return defaultRenderBody(pair, t);
		},
	};
}

/**
 * 所有差旅工具的渲染器，按**工具名**索引。
 *
 * 由父级 ``tool-renderers/index.tsx`` 展开进它的 ``renderers`` 表 ——
 * 那里是框架分发的真源，这里只是把差旅工具那一块**成组**放，避免把 7 个
 * 工具名散落进框架内建工具（Bash/Read/...）的清单里。
 */
export const ALIGO_RENDERERS: Record<string, ToolRenderer> = Object.fromEntries(
	// 显式标注元组类型：不标的话 TS 会把 ["name", renderer] 推成
	// (string | ToolRenderer)[] 而非 [string, ToolRenderer]，Object.fromEntries
	// 就只能退到返回 any 的重载上，等于这条类型约束白写了。
	ALIGO_TOOLS.map((tool): [string, ToolRenderer] => [tool.name, makeAligoRenderer(tool)]),
);
