import { CardShell, CardTitle, Field, Pill } from './shell';
import type { AligoResult } from './parse';
import { asArray, asNumber, asRecord, asString, formatMoney, itemRecords } from './parse';
import type { TFunction } from '../types';

/**
 * ``policy_verdict`` 卡片 —— 差旅标准核对结论。
 *
 * 数据来自 :mod:`src/tools/travel` 的 ``check_travel_policy``。
 *
 * ⚠️ 这张卡片承载**两种语义**，靠负载里的 ``lookup`` 字段区分：
 *
 * - ``lookup: true`` —— 用户问「标准是多少」，工具只回标准，**没有**
 *   ``compliant``。此时徽章必须是**中性**的。
 * - ``lookup: false`` —— 用户给了具体价格/舱位，工具给了核对结论，
 *   徽章按 ``compliant`` 出绿或红。
 *
 * ⚠️ 早先这里只写 ``compliant === true ? 绿 : 红``，于是查标准那条路的
 * ``undefined`` 被判成 false，用户问一句「住宿标准是多少」会看到一个
 * 红色「不符合」徽标 —— 一句无中生有的违规指控。这正是 2026-10-03
 * 实测到的缺陷，``lookup`` 字段就是为它加的。
 *
 * ⚠️ 合规与否**只用 ``compliant`` 布尔字段**判定，不通过解析 ``summary``
 * 文本里有没有「不符合」三个字来猜。summary 是给人读的自然语言，措辞随时
 * 可能调整；一旦用它当判据，改一句文案就会让卡片把「违规」显示成「合规」
 * —— 这正是 :mod:`src/tools/_result` 强调「``summary`` 与 ``card``/``items``
 * 分工」的原因。
 */
export function PolicyVerdictCard({ data, t }: { data: AligoResult; t: TFunction }) {
	const record = asRecord(itemRecords(data)[0]);
	if (Object.keys(record).length === 0) return null;

	// 三态，不是布尔：查标准 / 符合 / 不符合。
	const isLookup = record.lookup === true;
	const compliant = record.compliant === true;
	const pillTone: 'muted' | 'success' | 'danger' = isLookup ? 'muted' : compliant ? 'success' : 'danger';
	const pillText = isLookup
		? t('toolCard.policy.lookup')
		: compliant
			? t('toolCard.policy.ok')
			: t('toolCard.policy.violation');
	const reasons = asArray(record.reasons)
		.map((r) => asString(r))
		.filter(Boolean);
	const advice = asString(record.advice);
	const note = asString(record.policy_note);

	// 标准上限：两个都可能为 0（该差标未限制这一类），只显示真正有值的那些。
	const limits: string[] = [];
	const maxHotel = asNumber(record.max_hotel_price, 0);
	const maxFlight = asNumber(record.max_flight_price, 0);
	// ⚠️ 优先用后端给的中文名，别自己去查枚举表：前端反查原始枚举
	// （``ECONOMY`` → 「经济舱」）会让两份词表在前后端各存一份，
	// 后端加一个舱位而前端没跟上时，卡片直接显示英文码。
	const maxCabin = asString(record.max_cabin_text) || asString(record.max_cabin);
	if (maxHotel > 0) limits.push(`${t('toolCard.policy.hotelLimit')} ${formatMoney(maxHotel)}`);
	if (maxFlight > 0) limits.push(`${t('toolCard.policy.flightLimit')} ${formatMoney(maxFlight)}`);
	if (maxCabin) limits.push(`${t('toolCard.policy.cabinLimit')} ${maxCabin}`);

	return (
		<CardShell>
			<CardTitle>
				<span className="flex items-center gap-x-2">
					{t('toolCard.policy.title')}
					<Pill tone={pillTone}>{pillText}</Pill>
				</span>
			</CardTitle>
			{reasons.length > 0 && (
				<Field label={t('toolCard.policy.reasons')}>
					<ul className="list-disc pl-4">
						{reasons.map((reason, index) => (
							<li key={index} className="break-words">
								{reason}
							</li>
						))}
					</ul>
				</Field>
			)}
			{advice && <Field label={t('toolCard.policy.advice')}>{advice}</Field>}
			{limits.length > 0 && (
				<Field label={t('toolCard.policy.limit')}>
					<span className="text-muted-foreground">{limits.join(' · ')}</span>
				</Field>
			)}
			{note && (
				<div className="text-muted-foreground">
					{t('toolCard.policy.basis')}
					{note}
				</div>
			)}
		</CardShell>
	);
}
