import { CardShell, CardTitle } from './shell';
import type { AligoResult } from './parse';
import { asNumber, asString, formatMoney, itemRecords } from './parse';
import type { TFunction } from '../types';

/**
 * ``hotel_options`` 卡片 —— 酒店列表。
 *
 * 数据来自 :mod:`src/tools/travel` 的 ``search_hotels``。
 *
 * ⚠️ 与交通卡片一样，``items === []`` 是「这个城市/商圈没有酒店」的
 * **业务结论**，不是失败 —— 空列表照常渲染卡片并给出明确文案。
 */
export function HotelOptionsCard({ data, t }: { data: AligoResult; t: TFunction }) {
	const hotels = itemRecords(data);
	const city = asString(data.city);

	return (
		<CardShell>
			<CardTitle hint={city || undefined}>{t('toolCard.hotel.title')}</CardTitle>
			{hotels.length === 0 ? (
				<div className="text-muted-foreground">{t('toolCard.hotel.empty')}</div>
			) : (
				hotels.map((hotel, index) => {
					const star = asNumber(hotel.star, 0);
					const distance = asNumber(hotel.distance_km, 0);
					return (
						<div
							key={asString(hotel.option_id) || index}
							className="flex flex-col gap-y-1 border-t pt-1.5 first:border-t-0 first:pt-0"
						>
							<div className="flex items-baseline justify-between gap-x-2 min-w-0">
								<span className="min-w-0 truncate font-medium">
									{asString(hotel.name)}
								</span>
								<span className="whitespace-nowrap font-medium">
									{formatMoney(hotel.price_per_night)}
									<span className="text-muted-foreground font-normal">
										{t('toolCard.units.perNight')}
									</span>
								</span>
							</div>
							<div className="flex items-center justify-between gap-x-2 text-muted-foreground">
								<span className="min-w-0 truncate">
									{/* 星级与距离都是可选字段（可能为 0），为 0 时不显示，
									    避免出现「0★」「0km」这种看起来像真实数据的噪声。 */}
									{[asString(hotel.area), star > 0 ? `${star}★` : '']
										.filter(Boolean)
										.join(' · ')}
								</span>
								{distance > 0 && (
									<span className="whitespace-nowrap">
										{distance}
										{t('toolCard.units.km')}
									</span>
								)}
							</div>
						</div>
					);
				})
			)}
		</CardShell>
	);
}
