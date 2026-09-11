const http = require('http');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const PORT = 3000;

const MIME_TYPES = {
    '.html': 'text/html',
    '.js': 'text/javascript',
    '.wasm': 'application/wasm',
};

http.createServer((req, res) => {
    let filePath = '.' + req.url;
    if (filePath === './') filePath = './index.html';

    const extname = path.extname(filePath);
    const contentType = MIME_TYPES[extname] || 'application/octet-stream';

    fs.readFile(filePath, (error, content) => {
        if (error) {
            res.writeHead(404);
            res.end('File not found');
        } else {
            res.writeHead(200, {
                'Content-Type': contentType,
                'Cross-Origin-Opener-Policy': 'same-origin',
                'Cross-Origin-Embedder-Policy': 'require-corp'
            });
            res.end(content, 'utf-8');
        }
    });
}).listen(PORT, () => {
    const targetUrl = `http://localhost:${PORT}/`;
    console.log(`Isolated server running at ${targetUrl}`);

    const browserFlags = [
        '--enable-features=Vulkan,DefaultANGLEVulkan,VulkanFromANGLE',
        '--enable-unsafe-webgpu',
        targetUrl
    ];

    // Spawns Chromium on the RTX 3060 via prime-run
    const browser = spawn('prime-run', ['google-chrome-stable', ...browserFlags], {
        detached: true,
        stdio: 'ignore'
    });

    // Detaches child process so the terminal isn't tied to browser logging
    browser.unref();
});