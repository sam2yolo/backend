import { json } from '@sveltejs/kit';
import { createReadStream } from 'node:fs';
import { Readable } from 'node:stream';
import { getLocalMirror, mirrorRemoteUrl } from '$lib/server/localMirror';

export async function POST({ request }) {
	try {
		const payload = await request.json();
		const sourceUrl = String(payload.url || '').trim();
		if (!sourceUrl) return json({ message: 'Missing source URL' }, { status: 400 });

		const mirror = await mirrorRemoteUrl(sourceUrl, { kind: payload.kind || 'remote' });
		return json({ mirror });
	} catch (error) {
		return json(
			{ message: error instanceof Error ? error.message : 'Local mirror failed' },
			{ status: 502 }
		);
	}
}

export async function GET({ url, request }) {
	try {
		const mirror = await getLocalMirror(url.searchParams.get('id'));
		if (!mirror) return new Response('Local mirror not found', { status: 404 });

		const headers = new Headers({
			'accept-ranges': 'bytes',
			'content-type': mirror.contentType || 'application/octet-stream',
			'content-disposition': `inline; filename="${mirror.name.replaceAll('"', '')}"`
		});

		const range = request.headers.get('range');
		if (range) {
			const match = range.match(/bytes=(\d*)-(\d*)/);
			if (match) {
				const start = match[1] ? Number(match[1]) : 0;
				const end = match[2] ? Number(match[2]) : mirror.size - 1;
				const safeStart = Math.max(0, Math.min(start, mirror.size - 1));
				const safeEnd = Math.max(safeStart, Math.min(end, mirror.size - 1));
				headers.set('content-range', `bytes ${safeStart}-${safeEnd}/${mirror.size}`);
				headers.set('content-length', String(safeEnd - safeStart + 1));
				return new Response(
					Readable.toWeb(createReadStream(mirror.path, { start: safeStart, end: safeEnd })),
					{ status: 206, headers }
				);
			}
		}

		headers.set('content-length', String(mirror.size));
		return new Response(Readable.toWeb(createReadStream(mirror.path)), { status: 200, headers });
	} catch (error) {
		return new Response(error instanceof Error ? error.message : 'Local mirror fetch failed', {
			status: 502
		});
	}
}
