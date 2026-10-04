import type { ReactNode } from 'react';

import type { AligoResult } from './parse';
import type { TFunction } from '../types';
import { cn } from '@/lib/utils';

/**
 * 差旅卡片的外壳：带边框的浅色盒子，内部纵向排列。
 *
 * ⚠️ 抽出来统一用，而不是每张卡片各写一遍样式：卡片会越长越多（每加一个
 * 工具就多一张），样式各写各的时，边框圆角、内边距、字号会在几次迭代后
 * 悄悄分叉，最后同一屏里出现三种"卡片"，看起来像三个不同的组件拼的。
 * 与示例前端 `_shared.tsx` 里 FileBody 的取舍一致：**外壳归外壳，
 * 内容归内容**。
 */
export function CardShell({ children }: { children: ReactNode }) {
	return (
		<div className="flex flex-col gap-y-1.5 border rounded-sm bg-background p-2 text-xs">
			{children}
		</div>
	);
}

/** 卡片标题行：一个字重稍高的标题，可选地跟一段次要说明。 */
export function CardTitle({ children, hint }: { children: ReactNode; hint?: string }) {
	return (
		<div className="flex items-baseline justify-between gap-x-2">
			<span className="font-medium">{children}</span>
			{hint && <span className="text-muted-foreground">{hint}</span>}
		</div>
	);
}

/** 一行「标签: 值」；值缺失时整行不渲染，避免出现一排空标签。 */
export function Field({ label, children }: { label: string; children: ReactNode }) {
	if (children === null || children === undefined || children === '') return null;
	return (
		<div className="flex items-baseline gap-x-2 min-w-0">
			<span className="text-muted-foreground shrink-0">{label}</span>
			<span className="min-w-0 break-words">{children}</span>
		</div>
	);
}

/**
 * 小状态标（合规/不合规、已提交等）。
 *
 * ⚠️ 颜色**只**用来强化文字，不单独承载语义：标里始终有文字，色盲用户
 * 与深色模式下都能读；纯靠颜色区分合不合规的卡片在灰度打印里等于没标。
 */
export function Pill({
	tone = 'muted',
	children,
}: {
	tone?: 'success' | 'danger' | 'warn' | 'muted';
	children: ReactNode;
}) {
	const toneClass = {
		success: 'text-emerald-700 dark:text-emerald-400 bg-emerald-500/10',
		danger: 'text-red-700 dark:text-red-400 bg-red-500/10',
		warn: 'text-amber-700 dark:text-amber-400 bg-amber-500/10',
		muted: 'text-muted-foreground bg-muted',
	}[tone];
	return (
		<span className={cn('rounded-sm px-1.5 py-0.5 font-medium whitespace-nowrap', toneClass)}>
			{children}
		</span>
	);
}

/**
 * 「信息不足，需追问」与「调用失败」两种**非卡片**结果的统一展示。
 *
 * ⚠️ 这两种结果后端的 ``card`` 都是空串（见 ``src/tools/_result.py``），
 * 所以走不到任何卡片渲染器 —— 但它们又**确实**需要区别于「普通成功」的
 * 呈现：追问要让用户一眼看到还缺什么要素，失败要让用户知道该重试。
 * 若放任它们掉进默认渲染（一段裸 JSON 文本），用户会看到
 * ``{"ok":false,...}`` 这种东西，既难看又读不懂。
 *
 * ⚠️ **失败分支只渲染 `summary`，绝不要渲染 `detail`。**
 * `detail` 是内部标识与异常原文（``unknown kind='hotel'``、
 * ``TimeoutError: ...``，见 ``src/tools/_result.py::error_chunk``），
 * 它随载荷一路进到浏览器，但**不属于用户界面**：等宽字体展示一行
 * Python 异常既看不懂又像系统坏了。2026-10-03 的审计发现这里正是
 * 这么渲染的，已删除；服务端改为把 `detail` 写进日志。
 * 想加回来之前先看 `tests/test_frontend_contract.py` 里守着这条的用例。
 *
 * @param data - 已解析的差旅载荷。
 * @param t - i18n 取词函数。
 */
export function NoticeBody({ data, t }: { data: AligoResult; t: TFunction }) {
	if (data.ok === false) {
		return (
			<CardShell>
				<div className="flex items-center gap-x-2">
					<Pill tone="danger">{t('toolCard.notice.failed')}</Pill>
					<span className="min-w-0 break-words">{data.summary}</span>
				</div>
			</CardShell>
		);
	}

	// needs_input：不是错误，只是还没凑齐要素。用中性的提示色，不用红。
	return (
		<CardShell>
			<div className="flex items-center gap-x-2">
				<Pill tone="warn">{t('toolCard.notice.needMore')}</Pill>
				<span className="min-w-0 break-words">{data.summary}</span>
			</div>
			{data.needs?.length ? (
				<div className="flex flex-wrap gap-1">
					{data.needs.map((need) => (
						<Pill key={need} tone="muted">
							{need}
						</Pill>
					))}
				</div>
			) : null}
		</CardShell>
	);
}
