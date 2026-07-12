# Beacon Web UI

该目录包含 Beacon 的本地 Web 工作台与 Node.js API 服务：

- `server.mjs`：提供健康检查、示例题、运行控制、日志流和产物读取接口；
- `index.html`、`app.js`、`styles.css`：浏览器端界面；
- `assets/`：Logo 与页面图片资源。

请从项目根目录启动：

```bash
npm start
```

然后访问 `http://127.0.0.1:5173`。直接打开 `index.html` 只能查看静态页面，无法调用运行、恢复和产物接口。

题面导入支持 JSON、Markdown 和 TXT。PDF 需要先转换为文本；当前前端不会上传或解析 PDF。
