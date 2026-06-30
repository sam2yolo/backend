import { createHash } from 'node:crypto';
import { createReadStream, createWriteStream } from 'node:fs';
import { mkdir, readFile, rename, stat, unlink, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { spawn } from 'node:child_process';

const cacheRoot = path.resolve(process.cwd(), '.samtoyolo-cache', 'imports');
const inflight = new Map();

function cacheIdForSource(source) {
	return createHash('sha256').update(String(source)).digest('hex').slice(0, 32);
}

function assertCacheId(id) {
	if (!/^[a-f0-9]{16,64}$/i.test(String(id || ''))) throw new Error('Invalid local mirror id');
	return String(id);
}

function filePathFor(id) {
	return path.join(cacheRoot, `${assertCacheId(id)}.bin`);
}

function metaPathFor(id) {
	return path.join(cacheRoot, `${assertCacheId(id)}.json`);
}

function safeRemoteUrl(value) {
	const url = new URL(String(value || ''));
	if (!['http:', 'https:'].includes(url.protocol))
		throw new Error('Only HTTP(S) URLs can be mirrored');
	return url.toString();
}

function sanitizeName(name, fallback = 'source-video') {
	const cleaned = String(name || '')
		.split(/[\\/]/)
		.pop()
		.replace(/[^\w.\- ()]+/g, '_')
		.replace(/^_+|_+$/g, '');
	return cleaned || fallback;
}

function filenameFromUrl(value, fallback = 'source-video') {
	try {
		const url = new URL(value);
		const name = decodeURIComponent(url.pathname.split('/').filter(Boolean).pop() || '');
		return sanitizeName(name, fallback);
	} catch {
		return fallback;
	}
}

function filenameFromDisposition(header = '') {
	const utf = /filename\*=UTF-8''([^;]+)/i.exec(header);
	if (utf?.[1]) return sanitizeName(decodeURIComponent(utf[1]));
	const plain = /filename="?([^";]+)"?/i.exec(header);
	if (plain?.[1]) return sanitizeName(plain[1]);
	return '';
}

function decodeHtml(value = '') {
	return String(value)
		.replaceAll('&amp;', '&')
		.replaceAll('&quot;', '"')
		.replaceAll('&#39;', "'")
		.replaceAll('&lt;', '<')
		.replaceAll('&gt;', '>');
}

function htmlAttr(tag = '', name = '') {
	const match = new RegExp(`${name}=["']([^"']*)["']`, 'i').exec(tag);
	return match?.[1] ? decodeHtml(match[1]) : '';
}

function driveFileIdFromUrl(value) {
	try {
		const url = new URL(value);
		const queryId = url.searchParams.get('id');
		if (queryId) return queryId;
		const match = url.pathname.match(/\/d\/([^/]+)/);
		return match?.[1] || '';
	} catch {
		return '';
	}
}

function isGoogleDriveUrl(value) {
	try {
		const host = new URL(value).hostname.toLowerCase();
		return host === 'drive.google.com' || host.endsWith('.drive.google.com');
	} catch {
		return false;
	}
}

function googleDriveDownloadUrl(value) {
	const id = driveFileIdFromUrl(value);
	return id ? `https://drive.google.com/uc?export=download&id=${encodeURIComponent(id)}` : value;
}

function parseCookies(response) {
	const header = response.headers.get('set-cookie');
	if (!header) return '';
	return header
		.split(/,(?=[^;,]+=)/)
		.map((part) => part.split(';')[0])
		.filter(Boolean)
		.join('; ');
}

function extractDriveConfirmUrl(html, baseUrl, originalUrl) {
	const href = html.match(/href="([^"]*uc\?export=download[^"]+)"/i)?.[1];
	if (href) return new URL(href.replaceAll('&amp;', '&'), baseUrl).toString();

	const form = html.match(/<form[^>]*id=["']download-form["'][^>]*>[\s\S]*?<\/form>/i)?.[0];
	if (form) {
		const action = htmlAttr(form.match(/<form[^>]*>/i)?.[0] || '', 'action');
		const params = new URLSearchParams();
		for (const input of form.matchAll(/<input\b[^>]*>/gi)) {
			const name = htmlAttr(input[0], 'name');
			if (name) params.set(name, htmlAttr(input[0], 'value'));
		}
		if (action && params.get('confirm')) return `${new URL(action, baseUrl)}?${params.toString()}`;
	}

	const token = html.match(/[?&]confirm=([0-9A-Za-z_\-]+)/)?.[1];
	const id = driveFileIdFromUrl(originalUrl);
	if (token && id) {
		return `https://drive.google.com/uc?export=download&confirm=${encodeURIComponent(token)}&id=${encodeURIComponent(id)}`;
	}
	return '';
}

function extractDriveFileName(html) {
	const match = html.match(
		/<span[^>]*class=["']uc-name-size["'][^>]*>[\s\S]*?<a[^>]*>([^<]+)<\/a>/i
	);
	return match?.[1] ? sanitizeName(decodeHtml(match[1])) : '';
}

function publicMeta(meta) {
	return {
		cacheId: meta.cacheId,
		name: meta.name,
		size: meta.size,
		contentType: meta.contentType || 'application/octet-stream',
		sourceUrl: meta.sourceUrl || '',
		createdAt: meta.createdAt,
		method: meta.method || ''
	};
}

async function readMeta(cacheId) {
	try {
		const meta = JSON.parse(await readFile(metaPathFor(cacheId), 'utf8'));
		const info = await stat(filePathFor(cacheId));
		return publicMeta({ ...meta, cacheId, size: info.size });
	} catch {
		return null;
	}
}

async function writeMeta(cacheId, meta) {
	await mkdir(cacheRoot, { recursive: true });
	await writeFile(metaPathFor(cacheId), JSON.stringify({ ...meta, cacheId }, null, 2));
}

async function moveIntoCache(tmpPath, cacheId, meta) {
	await rename(tmpPath, filePathFor(cacheId));
	const info = await stat(filePathFor(cacheId));
	const record = {
		cacheId,
		name: sanitizeName(meta.name, `mirror-${cacheId}.bin`),
		size: info.size,
		contentType: meta.contentType || 'application/octet-stream',
		sourceUrl: meta.sourceUrl || '',
		createdAt: new Date().toISOString(),
		method: meta.method || ''
	};
	await writeMeta(cacheId, record);
	return publicMeta(record);
}

async function cleanupTemp(tmpPath) {
	try {
		await unlink(tmpPath);
	} catch {
		/* ignore */
	}
}

function runCommand(command, args) {
	return new Promise((resolve, reject) => {
		const child = spawn(command, args, { stdio: ['ignore', 'pipe', 'pipe'] });
		let output = '';
		child.stdout.on('data', (chunk) => {
			output += chunk.toString();
		});
		child.stderr.on('data', (chunk) => {
			output += chunk.toString();
		});
		child.on('error', reject);
		child.on('close', (code) => {
			if (code === 0) resolve(output);
			else reject(new Error(output.trim() || `${command} exited with status ${code}`));
		});
	});
}

async function downloadWithGdown(sourceUrl, tmpPath) {
	const id = driveFileIdFromUrl(sourceUrl);
	const targets = [...new Set([id, googleDriveDownloadUrl(sourceUrl), sourceUrl].filter(Boolean))];
	const commands = [
		(commandTarget) => ['python3', ['-m', 'gdown', '-O', tmpPath, commandTarget]],
		(commandTarget) => ['python', ['-m', 'gdown', '-O', tmpPath, commandTarget]],
		(commandTarget) => ['gdown', ['-O', tmpPath, commandTarget]]
	];
	const attempts = targets.flatMap((target) => commands.map((command) => command(target)));
	let lastError;
	for (const [command, args] of attempts) {
		try {
			await runCommand(command, args);
			const info = await stat(tmpPath);
			if (info.size > 0) return;
		} catch (error) {
			lastError = error;
			await cleanupTemp(tmpPath);
		}
	}
	throw lastError || new Error('gdown could not mirror Google Drive file');
}

async function downloadWithFetch(sourceUrl, tmpPath, kind = 'remote') {
	const fetchUrl =
		kind === 'drive' || isGoogleDriveUrl(sourceUrl) ? googleDriveDownloadUrl(sourceUrl) : sourceUrl;
	const headers = {
		'user-agent':
			'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) samtoyolo-local-mirror'
	};
	let response = await fetch(fetchUrl, { headers, redirect: 'follow' });
	let contentType = response.headers.get('content-type') || '';
	let driveFileName = '';

	if (response.ok && isGoogleDriveUrl(sourceUrl) && contentType.includes('text/html')) {
		const cookies = parseCookies(response);
		const html = await response.text();
		driveFileName = extractDriveFileName(html);
		const confirmUrl = extractDriveConfirmUrl(html, response.url, sourceUrl);
		if (!confirmUrl) throw new Error('Google Drive returned an HTML page without a download link');
		response = await fetch(confirmUrl, {
			headers: cookies ? { ...headers, cookie: cookies } : headers,
			redirect: 'follow'
		});
		contentType = response.headers.get('content-type') || '';
	}

	if (!response.ok) throw new Error(`Remote mirror failed with HTTP ${response.status}`);
	if (!response.body) throw new Error('Remote mirror response had no body');
	if (isGoogleDriveUrl(sourceUrl) && contentType.includes('text/html'))
		throw new Error('Google Drive returned an HTML page instead of the file download');

	const name =
		filenameFromDisposition(response.headers.get('content-disposition') || '') ||
		driveFileName ||
		filenameFromUrl(response.url || sourceUrl);
	await pipeline(Readable.fromWeb(response.body), createWriteStream(tmpPath));
	return {
		name,
		contentType: response.headers.get('content-type') || 'application/octet-stream',
		method: 'fetch'
	};
}

export async function storeUploadedMirror(file) {
	const sourceKey = `upload:${file.name}:${file.size}:${file.lastModified || ''}`;
	const cacheId = cacheIdForSource(sourceKey);
	const existing = await readMeta(cacheId);
	if (existing) return existing;

	await mkdir(cacheRoot, { recursive: true });
	const tmpPath = path.join(cacheRoot, `${cacheId}.${process.pid}.${Date.now()}.tmp`);
	try {
		await pipeline(Readable.fromWeb(file.stream()), createWriteStream(tmpPath));
		return await moveIntoCache(tmpPath, cacheId, {
			name: file.name,
			contentType: file.type || 'application/octet-stream',
			method: 'upload'
		});
	} catch (error) {
		await cleanupTemp(tmpPath);
		throw error;
	}
}

export async function mirrorRemoteUrl(sourceUrl, { kind = 'remote' } = {}) {
	const safeUrl = safeRemoteUrl(sourceUrl);
	const cacheId = cacheIdForSource(`remote:${safeUrl}`);
	const existing = await readMeta(cacheId);
	if (existing) return existing;
	if (inflight.has(cacheId)) return inflight.get(cacheId);

	const promise = (async () => {
		await mkdir(cacheRoot, { recursive: true });
		const tmpPath = path.join(cacheRoot, `${cacheId}.${process.pid}.${Date.now()}.tmp`);
		try {
			let meta;
			if (kind === 'drive' || isGoogleDriveUrl(safeUrl)) {
				try {
					await downloadWithGdown(safeUrl, tmpPath);
					meta = {
						name: filenameFromUrl(safeUrl, `google-drive-${cacheId}.bin`),
						contentType: 'application/octet-stream',
						sourceUrl: safeUrl,
						method: 'gdown'
					};
				} catch {
					meta = await downloadWithFetch(safeUrl, tmpPath, 'drive');
					meta.sourceUrl = safeUrl;
				}
			} else {
				meta = await downloadWithFetch(safeUrl, tmpPath, kind);
				meta.sourceUrl = safeUrl;
			}
			return await moveIntoCache(tmpPath, cacheId, meta);
		} catch (error) {
			await cleanupTemp(tmpPath);
			throw error;
		}
	})();

	inflight.set(cacheId, promise);
	try {
		return await promise;
	} finally {
		inflight.delete(cacheId);
	}
}

export async function getLocalMirror(cacheId) {
	const id = assertCacheId(cacheId);
	const meta = await readMeta(id);
	if (!meta) return null;
	return {
		...meta,
		path: filePathFor(id)
	};
}
