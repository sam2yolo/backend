// Client helpers for room/worker management via the /api/broker proxy.

const ADJECTIVES = [
	'nimble',
	'curious',
	'brave',
	'clever',
	'swift',
	'gentle',
	'bold',
	'calm',
	'eager',
	'fuzzy',
	'jolly',
	'lucky',
	'mighty',
	'quiet',
	'rapid',
	'shiny',
	'sleek',
	'sunny',
	'witty',
	'zesty',
	'cosmic',
	'fancy',
	'humble',
	'noble'
];
const ANIMALS = [
	'sloth',
	'otter',
	'falcon',
	'panda',
	'lynx',
	'heron',
	'badger',
	'koala',
	'marmot',
	'gecko',
	'walrus',
	'puffin',
	'tapir',
	'ibex',
	'raven',
	'mantis',
	'narwhal',
	'meerkat',
	'quokka',
	'civet',
	'beagle',
	'condor',
	'dingo',
	'wombat'
];

export const RUN_SH_URL =
	'https://raw.githubusercontent.com/sam2yolo/backend/refs/heads/main/run.sh';

function shellQuote(value) {
	return `'${String(value).replaceAll("'", "'\"'\"'")}'`;
}

function pythonString(value) {
	return JSON.stringify(String(value));
}

/** Reddit-style handle: adjective-animal-NNNN (e.g. nimble-sloth-9341). */
export function randomWorkerName() {
	const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];
	const num = Math.floor(1000 + Math.random() * 9000);
	return `${pick(ADJECTIVES)}-${pick(ANIMALS)}-${num}`;
}

/** The one-line command the user pastes on each GPU machine to join the room. */
export function bootstrapCommand(roomId, roomSecret, workerName) {
	return `cd ~ && wget -q -O run.sh "${RUN_SH_URL}" && bash run.sh ${shellQuote(
		roomId
	)} ${shellQuote(roomSecret)} ${shellQuote(workerName)}`;
}

/** Notebook cell that runs the generated terminal command through a pseudo-terminal. */
export function notebookBootstrapCell(roomId, roomSecret, workerName = 'samtoyolo-backend') {
	const command = bootstrapCommand(roomId, roomSecret, workerName || 'samtoyolo-backend');
	return [
		'# Notebook terminal bridge. Keep this cell running while using the website.',
		'import os',
		'import pty',
		'import select',
		'import signal',
		'import subprocess',
		'import sys',
		'import time',
		'',
		`COMMAND = ${pythonString(command)}`,
		'AUTO_REPLY_PATTERNS = (',
		'    "do you want to continue",',
		'    "[y/n]",',
		'    "(y/n)",',
		'    "[y/n]?",',
		'    "proceed ([y]/n)",',
		'    "proceed ([y]/n)?",',
		')',
		'',
		'def cleanup_tmux_sessions():',
		'    for session in ("s1", "sam-server"):',
		'        subprocess.run(["tmux", "kill-session", "-t", session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)',
		'',
		'env = os.environ.copy()',
		'env.setdefault("TERM", "xterm-256color")',
		'env.setdefault("DEBIAN_FRONTEND", "noninteractive")',
		'master_fd, slave_fd = pty.openpty()',
		'proc = subprocess.Popen(',
		'    ["/bin/bash", "-lc", COMMAND],',
		'    stdin=slave_fd,',
		'    stdout=slave_fd,',
		'    stderr=slave_fd,',
		'    env=env,',
		'    start_new_session=True,',
		'    close_fds=True,',
		')',
		'os.close(slave_fd)',
		'recent = ""',
		'print(f"[INFO] Running terminal command through a pseudo-terminal. pid={proc.pid}", flush=True)',
		'print(f"$ {COMMAND}", flush=True)',
		'try:',
		'    while True:',
		'        ready, _, _ = select.select([master_fd], [], [], 0.2)',
		'        if master_fd in ready:',
		'            try:',
		'                data = os.read(master_fd, 4096)',
		'            except OSError:',
		'                break',
		'            if not data:',
		'                break',
		'            text = data.decode("utf-8", "replace")',
		'            sys.stdout.write(text)',
		'            sys.stdout.flush()',
		'            recent = (recent + text)[-4000:]',
		'            lower = recent.lower()',
		'            if any(pattern in lower for pattern in AUTO_REPLY_PATTERNS):',
		'                os.write(master_fd, b"y\\n")',
		'                recent = ""',
		'        if proc.poll() is not None:',
		'            time.sleep(0.2)',
		'            while True:',
		'                ready, _, _ = select.select([master_fd], [], [], 0)',
		'                if master_fd not in ready:',
		'                    break',
		'                try:',
		'                    data = os.read(master_fd, 4096)',
		'                except OSError:',
		'                    break',
		'                if not data:',
		'                    break',
		'                sys.stdout.write(data.decode("utf-8", "replace"))',
		'                sys.stdout.flush()',
		'            break',
		'except KeyboardInterrupt:',
		'    print("\\n[INFO] Interrupt received; stopping the terminal process.", flush=True)',
		'    raise',
		'finally:',
		'    try:',
		'        if proc.poll() is None:',
		'            os.killpg(proc.pid, signal.SIGTERM)',
		'            try:',
		'                proc.wait(timeout=15)',
		'            except subprocess.TimeoutExpired:',
		'                os.killpg(proc.pid, signal.SIGKILL)',
		'    finally:',
		'        cleanup_tmux_sessions()',
		'        os.close(master_fd)',
		'if proc.returncode:',
		'    raise RuntimeError(f"Terminal command exited with status {proc.returncode}")'
	].join('\n');
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
