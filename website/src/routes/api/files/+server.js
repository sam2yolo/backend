function safeBackendUrl(value) {
	const url = new URL(value);
	if (!['http:', 'https:'].includes(url.protocol))
		throw new Error('Backend must use HTTP or HTTPS');
	return url.origin;
}

export async function GET({ url }) {
	try {
		const id = url.searchParams.get('id');
		if (!id) return new Response('Missing file id', { status: 400 });

		const backendUrl = safeBackendUrl(url.searchParams.get('backend') || 'http://127.0.0.1:8000');
		const response = await fetch(`${backendUrl}/file?id=${encodeURIComponent(id)}`);

		if (!response.ok) {
			const message = await response.text();
			return new Response(message || 'Backend file fetch failed', { status: response.status });
		}

		const headers = new Headers();
		const contentType = response.headers.get('content-type');
		const contentLength = response.headers.get('content-length');
		const contentDisposition = response.headers.get('content-disposition');
		if (contentType) headers.set('content-type', contentType);
		if (contentLength) headers.set('content-length', contentLength);
		if (contentDisposition) headers.set('content-disposition', contentDisposition);

		return new Response(response.body, { status: 200, headers });
	} catch (error) {
		return new Response(error instanceof Error ? error.message : 'Backend file fetch failed', {
			status: 502
		});
	}
}
