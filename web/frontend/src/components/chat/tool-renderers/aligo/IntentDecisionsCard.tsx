import { CardShell, CardTitle, Pill } from './shell';
import type { AligoResult } from './parse';
import { asNumber, asString, itemRecords } from './parse';
import type { TFunction } from '../types';

/**
 * ``recognize_intent`` 的结果展示 —— 它**没有卡片**（后端的 ``card`` 是空串）。
 *
 * 之所以还要专门为它写一段渲染：它的载荷里带着 ``reasoning``（识别过程的
 * 推理）、``rewritten_query``（改写后的检索语句）、``needs_clarification``
 * （是否需要追问）这些**给人看很有价值、但默认渲染只会当 JSON 文本吐出来**
 * 的字段。工具卡片体系按 ``card`` 分发，它落不进任何一类卡片；于是这里按
 * **工具名**给它一个兜底视图。
 *
 * ⚠️ 前提是结果里确实有 ``items``。空 items 时返回 ``null``，交回默认渲染
 * （显示 summary 文本）—— 因为这时没有可结构化的东西可画。
 */
export function IntentDecisionsCard({ data, t }: { data: AligoResult; t: TFunction }) {
	const decisions = itemRecords(data);
	if (decisions.length === 0) return null;

	const reasoning = asString(data.reasoning);
	const rewritten = asString(data.rewritten_query);
	const needsClarification = data.needs_clarification === true;
	const question = asString(data.clarification_question);

	return (
		<CardShell>
			<CardTitle>{t('toolCard.intent.title')}</CardTitle>
			{decisions.map((decision, index) => {
				const intent = asString(decision.intent);
				const confidence = asNumber(decision.confidence, 0);
				const reason = asString(decision.reason);
				return (
					<div
						key={`${intent}-${index}`}
						className="flex flex-col gap-y-0.5 border-t pt-1.5 first:border-t-0 first:pt-0"
					>
						<div className="flex items-baseline justify-between gap-x-2">
							<span className="font-medium">{intent}</span>
							<span className="text-muted-foreground whitespace-nowrap">
								{t('toolCard.intent.confidence')} {confidence.toFixed(2)}
							</span>
						</div>
						{reason && (
							<div className="text-muted-foreground break-words">{reason}</div>
						)}
					</div>
				);
			})}
			{reasoning && (
				<div className="border-t pt-1.5 text-muted-foreground">
					<span className="font-medium">{t('toolCard.intent.reasoning')}</span>
					<div className="mt-0.5 break-words">{reasoning}</div>
				</div>
			)}
			{rewritten && (
				<div className="text-muted-foreground break-words">
					{t('toolCard.intent.rewritten')}
					<span className="font-mono">{rewritten}</span>
				</div>
			)}
			{needsClarification && (
				<div className="flex items-center gap-x-2 border-t pt-1.5">
					<Pill tone="warn">{t('toolCard.intent.clarify')}</Pill>
					{question && <span className="min-w-0 break-words">{question}</span>}
				</div>
			)}
		</CardShell>
	);
}
