// Client helpers for room/worker management via the /api/broker proxy.

const ADJECTIVES = [
	'nimble', 'curious', 'brave', 'clever', 'swift', 'gentle', 'bold', 'calm',
	'eager', 'fuzzy', 'jolly', 'lucky', 'mighty', 'quiet', 'rapid', 'shiny',
	'sleek', 'sunny', 'witty', 'zesty', 'cosmic', 'fancy', 'humble', 'noble'
];
const ANIMALS = [
	'sloth', 'otter', 'falcon', 'panda', 'lynx', 'heron', 'badger', 'koala',
	'marmot', 'gecko', 'walrus', 'puffin', 'tapir', 'ibex', 'raven', 'mantis',
	'narwhal', 'meerkat', 'quokka', 'civet', 'beagle', 'condor', 'dingo', 'wombat'
];

export const RUN_SH_URL =
	'https://raw.githubusercontent.com/sam2yolo/backend/refs/heads/main/run.sh';

/** Reddit-style handle: adjective-animal-NNNN (e.g. nimble-sloth-9341). */
export function randomWorkerName() {
	const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];
	const num = Math.floor(1000 + Math.random() * 9000);
	return `${pick(ADJECTIVES)}-${pick(ANIMALS)}-${num}`;
}

/** The one-line command the user pastes on each GPU machine to join the room. */
export function bootstrapCommand(roomId, roomSecret, workerName) {
	return `wget "${RUN_SH_URL}" && bash run.sh ${roomId} ${roomSecret} ${workerName}`;
}

async function call(op, extra = {}) {
	const response = await fetch('/api/broker', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ op, ...extra })
	});
	const data = await response.json().catch(() => ({}));
	if (!response.ok) throw new Error(data.message || `Broker ${op} failed`);
	return data;
}

/** Create a room. Returns { room_id, room_secret, status, created_at }. */
export function createRoom(name = 'fleet-vision') {
	return call('create_room', { name });
}

/** List workers (tunnels) in a room. Returns an array of worker objects with
 *  http_url / ws_url already resolved. */
export async function listWorkers(roomId) {
	const data = await call('list_workers', { room_id: roomId });
	return data.workers || [];
}
