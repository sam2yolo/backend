function safeBackendUrl(value) {
	const url = new URL(value);
	if (!['http:', 'https:'].includes(url.protocol))
		throw new Error('Backend must use HTTP or HTTPS');
	return url.origin;
}

export async function GET({ url }) {
	try {
		const id = url.searchParams.get('id');
		if (!id) return new Response('Missing result id', { status: 400 });

		const backendUrl = safeBackendUrl(url.searchParams.get('backend') || 'http://127.0.0.1:8000');
		const response = await fetch(`${backendUrl}/inference_result?id=${encodeURIComponent(id)}`);

		if (!response.ok) {
			const message = await response.text();
			return new Response(message || 'Backend result fetch failed', { status: response.status });
		}

		const headers = new Headers();
		headers.set('content-type', response.headers.get('content-type') || 'application/octet-stream');
		headers.set('content-disposition', `attachment; filename="inference-${id}.pkl"`);

		return new Response(response.body, { status: 200, headers });
	} catch (error) {
		return new Response(error instanceof Error ? error.message : 'Backend result fetch failed', {
			status: 502
		});
	}
}
