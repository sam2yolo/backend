import { json } from '@sveltejs/kit';

// Server-side broker proxy. Keeps the API token out of the browser bundle and
// sidesteps CORS — the SvelteKit Node server (running on the user's machine)
// talks to the broker; the browser only talks to this route.
const BROKER = process.env.BROKER_URL || 'http://163.61.236.112:7001';
const API_TOKEN = process.env.BROKER_API_TOKEN || 'VKl7VeOSKDBLPheW7hdKyZQ5iq//B+1SSBXCkQpZ9HQ=';
const TUNNEL_HOST = process.env.TUNNEL_HOST || '163.61.236.112';

async function broker(path, options = {}) {
	const response = await fetch(`${BROKER}${path}`, {
		...options,
		headers: {
			'X-API-Token': API_TOKEN,
			'Content-Type': 'application/json',
			...(options.headers || {})
		}
	});
	const text = await response.text();
	let data;
	try {
		data = text ? JSON.parse(text) : {};
	} catch {
		data = { raw: text };
	}
	if (!response.ok) throw new Error(data.error || data.message || `Broker error ${response.status}`);
	return data;
}

function shapeWorker(tunnel) {
	// tunnel.name looks like "nimble-sloth-9341-8000" (worker name + local port)
	const name = String(tunnel.name || '').replace(/-(\d+)$/, '');
	return {
		tunnel_id: tunnel.tunnel_id,
		name,
		full_name: tunnel.name,
		remote_port: tunnel.remote_port,
		status: tunnel.status,
		alive: tunnel.is_alive ?? null,
		heartbeat_at: tunnel.heartbeat_at ?? null,
		http_url: `http://${TUNNEL_HOST}:${tunnel.remote_port}`,
		ws_url: `ws://${TUNNEL_HOST}:${tunnel.remote_port}/ws`
	};
}

export async function POST({ request }) {
	try {
		const body = await request.json();
		const op = body.op;

		if (op === 'create_room') {
			const data = await broker('/rooms', {
				method: 'POST',
				body: JSON.stringify({ name: body.name || 'fleet-vision' })
			});
			return json(data); // { room_id, room_secret, status, created_at }
		}

		if (op === 'list_workers') {
			if (!body.room_id) return json({ message: 'room_id is required' }, { status: 400 });
			const data = await broker(`/rooms/${encodeURIComponent(body.room_id)}/tunnels`);
			return json({
				room_id: data.room_id,
				workers: (data.tunnels || []).map(shapeWorker)
			});
		}

		return json({ message: `Unknown op: ${op}` }, { status: 400 });
	} catch (error) {
		return json(
			{ message: error instanceof Error ? error.message : 'Broker request failed' },
			{ status: 502 }
		);
	}
}
