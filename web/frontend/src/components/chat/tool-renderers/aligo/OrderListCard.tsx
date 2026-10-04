import { CardShell, CardTitle, Pill } from './shell';
import type { AligoResult } from './parse';
import { asString, formatMoney, itemRecords } from './parse';
import type { TFunction } from '../types';

/**
 * ``order_list`` 卡片 —— 订单与出差申请单的混合列表。
 *
 * 数据来自 :mod:`src/tools/orders` 的 ``query_orders``。
 *
 * ⚠️ 同一张卡片里混着两种条目（订单 ``type="order"``、申请单
 * ``type="approval"``），字段并不相同。这里按 ``type`` 分支渲染，**不**假设
 * 两者都有 ``amount``/``title`` —— 缺字段时 :func:`asString` 会给出空串，
 * 行内相应部分自然消失。若硬套同一套字段，缺的那半会显示成 ``undefined``。
 *
 * ⚠️ 用条目自身的业务主键（``order_id`` / ``request_id``）当 key，不用下标：
 * 列表是从两个数据源拼接出来的，顺序在重新查询后可能变化，下标做 key 会让
 * React 复用错误的行。缺失主键时才退回 `type`+下标组合。
 */
export function OrderListCard({ data, t }: { data: AligoResult; t: TFunction }) {
	const entries = itemRecords(data);

	return (
		<CardShell>
			<CardTitle>{t('toolCard.orders.title')}</CardTitle>
			{entries.length === 0 ? (
				<div className="text-muted-foreground">{t('toolCard.orders.empty')}</div>
			) : (
				entries.map((entry, index) => {
					const isApproval = asString(entry.type) === 'approval';
					const key =
						asString(isApproval ? entry.request_id : entry.order_id) ||
						`${asString(entry.type)}-${index}`;
					const created = asString(entry.created_at);
					return (
						<div
							key={key}
							className="flex flex-col gap-y-1 border-t pt-1.5 first:border-t-0 first:pt-0"
						>
							<div className="flex items-baseline justify-between gap-x-2 min-w-0">
								<span className="min-w-0 truncate">
									<Pill tone="muted">
										{isApproval
											? t('toolCard.orders.approval')
											: t('toolCard.orders.order')}
									</Pill>
									<span className="ml-1.5 font-medium">
										{asString(entry.title)}
									</span>
								</span>
								<span className="whitespace-nowrap font-medium">
									{formatMoney(entry.amount)}
								</span>
							</div>
							<div className="flex items-center justify-between gap-x-2 text-muted-foreground">
								<span className="min-w-0 truncate">
									{[
										asString(entry.status),
										asString(entry.destination),
										asString(entry.depart_date),
									]
										.filter(Boolean)
										.join(' · ')}
								</span>
								<span className="whitespace-nowrap">{created}</span>
							</div>
						</div>
					);
				})
			)}
		</CardShell>
	);
}
