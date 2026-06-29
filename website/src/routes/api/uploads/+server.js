import { json } from '@sveltejs/kit';

function safeBackendUrl(value) {
	const url = new URL(value);
	if (!['http:', 'https:'].includes(url.protocol))
		throw new Error('Backend must use HTTP or HTTPS');
	return url.origin;
}

export async function POST({ request }) {
	try {
		const incoming = await request.formData();
		const files = incoming
			.getAll('files')
			.filter((file) => typeof file?.arrayBuffer === 'function' && typeof file?.name === 'string');
		const backendUrl = safeBackendUrl(
			String(incoming.get('backendUrl') || 'http://127.0.0.1:8000')
		);

		if (!files.length) return json({ message: 'No files were uploaded' }, { status: 400 });

		const uploaded = [];
		for (const file of files) {
			const body = new FormData();
			body.append('file', file, file.name);
			const response = await fetch(`${backendUrl}/upload`, { method: 'POST', body });
			const result = await response.json();
			if (!response.ok || !result.file_id)
				throw new Error(result.error || `Backend rejected ${file.name}`);
			uploaded.push({
				id: result.file_id,
				name: file.name,
				path: `files/${result.file_id}`,
				size: file.size
			});
		}

		return json({ files: uploaded });
	} catch (error) {
		return json(
			{ message: error instanceof Error ? error.message : 'Backend upload failed' },
			{ status: 502 }
		);
	}
}
