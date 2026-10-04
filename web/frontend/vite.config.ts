import path from 'path';

import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import svgr from 'vite-plugin-svgr';

export default defineConfig({
	// ⚠️ base 必须是 "/"，不能用默认的相对路径 "./"。
	// 后端（src/server/app.py 的 _mount_static_if_present）把整个产物目录挂在
	// 根路径 "/" 上；前端路由是 /chat、/orders 这类深路径，刷新时浏览器按
	// 当前地址解析资源。base 若是 "./"（相对），在 /chat/123 这种页面上
	// 就会去请求 /chat/assets/index-xxx.js —— 而挂载点是根，那个路径下没有
	// 文件，页面直接白屏，且服务端日志里只有一串 404，看不出是 base 写错了。
	base: '/',
	plugins: [react(), tailwindcss(), svgr()],
	server: {
		proxy: {
			'/api': 'http://localhost:3000',
		},
	},
	resolve: {
		alias: {
			'@': path.resolve(__dirname, './src'),
			'next/navigation': path.resolve(__dirname, './src/lib/next-navigation-shim.ts'),
		},
	},
	build: {
		// ⚠️ 本任务最关键的一处配置：产物**必须**落到后端挂载的目录。
		// vite 默认输出到 frontend/dist，而后端只挂载 src/server/static/
		// （见 src/server/app.py 的 STATIC_DIR_NAME），两者对不上时
		// `pnpm build` 会「成功」，但后端 / 返回的仍是上一版（或仓库自带的
		// 占位页）—— 构建白做，且没有任何报错提示你走错了地方。
		// outDir 相对**工程根**（即 web/frontend/）解析：
		//   web/frontend/ → ../.. = 仓库根 → src/server/static/  ✅
		outDir: '../../src/server/static',
		// ⚠️ emptyOutDir 必须显式打开。outDir 在工程根之外时，vite 默认
		// **不清空**目标目录（只警告一句），于是每次构建的 hash 产物会
		// 层层堆积，且仓库自带的占位 index.html 会残留 —— 用户刷新看到的
		// 可能是旧页面。显式清空才能保证「构建产物」是这个目录里唯一的东西。
		emptyOutDir: true,
	},
	optimizeDeps: {
		include: ['mime-types'],
	},
});
