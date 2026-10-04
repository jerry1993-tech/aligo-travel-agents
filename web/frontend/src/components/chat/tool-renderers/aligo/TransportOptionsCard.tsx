import { CardShell, CardTitle, Pill } from './shell';
import type { AligoResult } from './parse';
import { asNumber, asString, formatDuration, formatMoney, itemRecords } from './parse';
import type { TFunction } from '../types';

/**
 * ``transport_options`` 卡片 —— 航班/火车选项列表。
 *
 * 数据来自 :mod:`src/tools/travel` 的 ``search_transport``。
 *
 * ⚠️ ``items === []``（查到了但没有班次）与 ``items === null``（这个工具
 * 不产出列表）是**两回事**，后端刻意区分（见 ``src/tools/_result.py`` 的
 * ``ok_chunk``）。这里用「有没有条目」判空，并为「查到空列表」给一句明确
 * 的文案 —— 它是一条**业务结论**（这条线路没有车/机），不是失败，
 * 所以卡片照常渲染，只是内容为空。
 */
export function TransportOptionsCard({ data, t }: { data: AligoResult; t: TFunction }) {
	const options = itemRecords(data);
	const origin = asString(data.origin);
	const destination = asString(data.destination);
	const departDate = asString(data.depart_date);
	const hint = [origin && destination ? `${origin} → ${destination}` : '', departDate]
		.filter(Boolean)
		.join(' · ');

	return (
		<CardShell>
			<CardTitle hint={hint || undefined}>{t('toolCard.transport.title')}</CardTitle>
			{options.length === 0 ? (
				<div className="text-muted-foreground">{t('toolCard.transport.empty')}</div>
			) : (
				options.map((option, index) => {
					const soldOut = option.sold_out === true;
					const seatsLeft = asNumber(option.seats_left, 0);
					return (
						<div
							// option_id 是后端保证的业务主键；缺失时退回下标，避免 React
							// 用下标当 key 造成的错位（列表顺序可能随价格变化）。
							key={asString(option.option_id) || index}
							className="flex flex-col gap-y-1 border-t pt-1.5 first:border-t-0 first:pt-0"
						>
							<div className="flex items-baseline justify-between gap-x-2 min-w-0">
								<span className="min-w-0 truncate">
									<span className="font-medium">{asString(option.carrier)}</span>
									<span className="text-muted-foreground">
										{' '}
										{asString(option.mode_display)}
									</span>
								</span>
								<span className="font-medium whitespace-nowrap">
									{formatMoney(option.price)}
								</span>
							</div>
							<div className="flex items-center justify-between gap-x-2 text-muted-foreground">
								<span className="min-w-0 truncate">
									{asString(option.depart_at)} → {asString(option.arrive_at)}
								</span>
								<span className="whitespace-nowrap">
									{formatDuration(option.duration_minutes)}
								</span>
							</div>
							<div className="flex items-center justify-between gap-x-2">
								<span className="text-muted-foreground">
									{asString(option.cabin)}
								</span>
								{soldOut ? (
									<Pill tone="danger">{t('toolCard.transport.soldOut')}</Pill>
								) : (
									<Pill tone="muted">
										{/* 参数名用 n 而不是 count：i18next 见到 count 会去按
										    复数规则找 seatsLeft_one/_other，而这里只定义了
										    不带后缀的一条，多绕一层解析。 */}
										{t('toolCard.transport.seatsLeft', { n: seatsLeft })}
									</Pill>
								)}
							</div>
						</div>
					);
				})
			)}
		</CardShell>
	);
}
