import { CardShell, CardTitle, Field, Pill } from './shell';
import type { AligoResult } from './parse';
import { asNumber, asRecord, asString, formatMoney, itemRecords } from './parse';
import type { TFunction } from '../types';

/**
 * ``approval_result`` 卡片 —— 提交出差申请单的**回执**。
 *
 * 数据来自 :mod:`src/tools/orders` 的 ``submit_approval``。这是一个**有副
 * 作用**的操作，前端在它执行前会先弹 ``ConfirmCard`` 让用户确认；本卡片是
 * 确认之后的回执，所以它要把「生成了哪张单、处于什么状态」说清楚，让用户
 * 能拿单号去追问或撤销。
 *
 * ⚠️ 单号（``request_id``）用等宽字体且**不截断**：它是用户唯一能在后续
 * 对话里指认这张单的凭据，被 CSS 截成 ``AP-7fK2…`` 就等于没给。
 */
export function ApprovalResultCard({ data, t }: { data: AligoResult; t: TFunction }) {
	const record = asRecord(itemRecords(data)[0]);
	if (Object.keys(record).length === 0) return null;

	const requestId = asString(record.request_id);
	const status = asString(record.status);
	const days = asNumber(record.days, 0);

	return (
		<CardShell>
			<CardTitle hint={status || undefined}>
				<span className="flex items-center gap-x-2">
					{t('toolCard.approval.title')}
					<Pill tone="success">{t('toolCard.approval.submitted')}</Pill>
				</span>
			</CardTitle>
			<Field label={t('toolCard.approval.requestId')}>
				<span className="font-mono break-all">{requestId}</span>
			</Field>
			<Field label={t('toolCard.approval.subject')}>{asString(record.title)}</Field>
			<Field label={t('toolCard.approval.destination')}>{asString(record.destination)}</Field>
			<Field label={t('toolCard.approval.depart')}>
				{[asString(record.depart_date), days > 0 ? `${days}${t('toolCard.units.day')}` : '']
					.filter(Boolean)
					.join(' · ')}
			</Field>
			<Field label={t('toolCard.approval.amount')}>{formatMoney(record.amount)}</Field>
		</CardShell>
	);
}
