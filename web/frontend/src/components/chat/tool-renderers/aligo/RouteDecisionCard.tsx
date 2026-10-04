import { CardShell, CardTitle, Field, Pill } from './shell';
import type { AligoResult } from './parse';
import { asArray, asRecord, asString, itemRecords } from './parse';
import type { TFunction } from '../types';

/**
 * ``route_decision`` 卡片 —— 快车道路由决策的可视化。
 *
 * 数据来自 :mod:`src/tools/route`（``aligo_route_intent``）：它把「系统已经
 * 判定用户意图是什么」显式告诉模型，本卡片则把同一件事告诉**用户**。
 *
 * ⚠️ 这里渲染 ``items[0]`` 而**不是**遍历整个 ``items``：路由决策天然只有
 * 一条（一次点击只有一个意图）。后端也用 ``items=[{...}]`` 表达这一点。
 * 若将来出现多条，说明路由模型变了 —— 那时应回到后端改契约，而不是让
 * 前端悄悄丢掉第二条。
 *
 * ⚠️ **卡片里有意不显示「下一步」。** 2026-10-03 的对抗审计发现：后端曾把
 * ``_NEXT_STEP_HINTS`` 同时塞进 ``items.next_step``，而这段文字是**写给
 * 模型的第二人称祈使句**（「请先调用 check_travel_policy 拿到标准数值」
 * 「在拿到结果之前不要回答」），于是用户在卡片上读到一句对他的助手说的
 * 指令。现在 ``next_step`` 字段已从 ``items`` 移除，指引只留在顶层
 * ``instruction``；``toolCard.route.next`` 这条 i18n 词条也一并删除。
 * 想加回来之前先看 ``tests/test_frontend_contract.py`` 里守着这条的用例 ——
 * 要显示给用户的「下一步」得是另一句话（面向用户、不含工具名），
 * 而不是把模型指引直接搬过来。
 */
export function RouteDecisionCard({ data, t }: { data: AligoResult; t: TFunction }) {
	const record = asRecord(itemRecords(data)[0]);
	if (Object.keys(record).length === 0) return null;

	const intent = asString(record.intent);
	const intentDisplay = asString(record.intent_display, intent);
	const rule = asString(record.matched_rule);
	const reason = asString(record.reason);
	const agents = asArray(record.target_agents)
		.map((a) => asString(a))
		.filter(Boolean);

	return (
		<CardShell>
			<CardTitle>{t('toolCard.route.title')}</CardTitle>
			<Field label={t('toolCard.route.intent')}>
				<span className="font-medium">{intentDisplay}</span>
				{intent && intentDisplay !== intent ? (
					<span className="text-muted-foreground"> ({intent})</span>
				) : null}
			</Field>
			<Field label={t('toolCard.route.rule')}>{rule}</Field>
			{agents.length > 0 && (
				<Field label={t('toolCard.route.agents')}>
					<span className="flex flex-wrap gap-1">
						{agents.map((agent) => (
							<Pill key={agent} tone="muted">
								{agent}
							</Pill>
						))}
					</span>
				</Field>
			)}
			{reason && (
				<Field label={t('toolCard.route.reason')}>
					<span className="text-muted-foreground">{reason}</span>
				</Field>
			)}
		</CardShell>
	);
}
