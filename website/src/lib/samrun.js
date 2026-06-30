// Run + distribute SAM 3.1 jobs across one or more workers.
//
// Each worker has its own WebSocket and a single model handler, so we open one
// connection per worker, init SAM, queue one task per prompt over that worker's
// shard of frames, and stream decoded results back. Frame indices are mapped
// from shard-local back to global so merged results stay aligned.

/**
 * Run a SAM job on a single worker.
 *
 * @param {object} opts
 * @param {string} opts.wsUrl            ws://host:port/ws
 * @param {string} opts.samBaseUrl       SAM server URL (backend-side)
 * @param {string[]} opts.fileIds        shard of backend file ids (images)
 * @param {number[]} opts.globalIndices  global frame index for each fileId
 * @param {string[]} opts.prompts        text prompts (class per prompt)
 * @param {number} opts.conf
 * @param {number} opts.batch
 * @param {(e:object)=>void} opts.onEvent  status / frame / error / done events
 * @returns {{ promise:Promise<void>, cancel:()=>void }}
 */
export function runSamOnWorker(opts) {
	const {
		wsUrl,
		samBaseUrl,
		fileIds,
		globalIndices,
		prompts,
		conf = 0.5,
		batch = 4,
		workerName = wsUrl,
		onEvent = () => {}
	} = opts;

	let ws;
	let cancelled = false;
	const taskMeta = new Map(); // task_id -> { className, classId }
	let expectedCompletions = prompts.length;
	let completions = 0;
	let pendingFetches = 0; // outstanding fetch_inference_chunk requests
	let queueDone = false;
	let done = false;

	const promise = new Promise((resolve, reject) => {
		try {
			ws = new WebSocket(wsUrl);
		} catch (error) {
			reject(error);
			return;
		}
		const send = (action, payload = {}) => ws.send(JSON.stringify({ action, payload }));
		const finish = (err) => {
			if (done) return;
			done = true;
			try {
				ws.close();
			} catch {
				/* ignore */
			}
			if (err) reject(err);
			else resolve();
		};
		// Only finish once all tasks are done AND every requested chunk has been
		// fetched (chunk_data arrives asynchronously, after inference_completed).
		const maybeFinish = () => {
			if ((queueDone || completions >= expectedCompletions) && pendingFetches <= 0) finish();
		};

		ws.addEventListener('open', () => {
			onEvent({ type: 'status', worker: workerName, message: 'Initializing SAM' });
			send('init_model', { model_name: 'sam', base_url: samBaseUrl });
		});

		ws.addEventListener('error', () => {
			if (!done) finish(new Error(`WebSocket error (${workerName})`));
		});
		ws.addEventListener('close', () => {
			// If the socket closed before we finished cleanly, resolve anyway so the
			// distributor isn't blocked (partial results already streamed).
			if (!done) {
				done = true;
				resolve();
			}
		});

		ws.addEventListener('message', (event) => {
			let msg;
			try {
				msg = JSON.parse(event.data);
			} catch {
				return;
			}
			const { action, payload = {} } = msg;
			switch (action) {
				case 'model_init_completed': {
					// Queue one task per prompt, then start the queue.
					prompts.forEach((prompt) => {
						send('create_inference_task', {
							file_ids: fileIds,
							file_type: 'image',
							text_prompt: prompt,
							conf,
							batch
						});
					});
					send('start_inference_from_queue');
					break;
				}
				case 'model_init_error':
				case 'model_setup_error':
					finish(new Error(payload.error || 'SAM init failed'));
					break;
				case 'task_added':
					taskMeta.set(payload.id, {
						className: payload.text_prompt,
						classId: prompts.indexOf(payload.text_prompt)
					});
					break;
				case 'inference_stage_plus_progress':
					onEvent({
						type: 'progress',
						worker: workerName,
						taskId: payload.task_id,
						progress: payload.progress,
						stage: payload.stage
					});
					break;
				case 'inference_task_chunk_result':
					// Pull the chunk JSON so we can render it.
					pendingFetches += 1;
					send('fetch_inference_chunk', {
						task_id: payload.task_id,
						chunk_id: payload.chunk_id
					});
					break;
				case 'inference_chunk_data': {
					const meta = taskMeta.get(payload.task_id) || {};
					const images = payload.data?.images || [];
					const chunkIndex = payload.chunk_index ?? 0;
					images.forEach((result, j) => {
						const localIdx = chunkIndex * batch + j;
						const globalIdx = globalIndices ? globalIndices[localIdx] : localIdx;
						onEvent({
							type: 'frame',
							worker: workerName,
							taskId: payload.task_id,
							className: meta.className,
							classId: meta.classId,
							frameIndex: globalIdx,
							fileId: fileIds[localIdx],
							result
						});
					});
					pendingFetches -= 1;
					maybeFinish();
					break;
				}
				case 'inference_chunks_error':
					pendingFetches -= 1;
					maybeFinish();
					break;
				case 'inference_completed':
					completions += 1;
					maybeFinish();
					break;
				case 'queue_completed':
					queueDone = true;
					maybeFinish();
					break;
				case 'inference_task_error':
				case 'task_failed':
					onEvent({
						type: 'error',
						worker: workerName,
						message: payload.error || 'inference error'
					});
					completions += 1;
					maybeFinish();
					break;
			}
		});
	});

	return {
		promise,
		cancel() {
			cancelled = true;
			try {
				ws?.close();
			} catch {
				/* ignore */
			}
		}
	};
}

/** Split N items into `parts` contiguous, roughly equal shards of indices. */
export function shardIndices(count, parts) {
	const shards = Array.from({ length: parts }, () => []);
	for (let i = 0; i < count; i++) shards[i % parts].push(i);
	return shards.filter((s) => s.length);
}

/**
 * Distribute a SAM job across workers. fileIds[i] corresponds to global frame i.
 * @param {Array<{ws_url:string,name?:string}>} workers
 */
export function distributeSam(workers, opts) {
	const { fileIds, assignments, prompts, samBaseUrl, conf, batch, onEvent } = opts;

	// Preferred path: explicit per-worker assignments. Each worker's files were
	// uploaded to THAT worker, so fileIds are only valid on their own worker.
	// assignments: [{ ws_url, name, fileIds:[...], globalIndices:[...] }]
	if (assignments && assignments.length) {
		const runs = assignments.map((a) => {
			onEvent?.({
				type: 'status',
				worker: a.name || a.ws_url,
				message: `Assigned ${a.fileIds.length} frame(s)`
			});
			return runSamOnWorker({
				wsUrl: a.ws_url,
				samBaseUrl,
				fileIds: a.fileIds,
				globalIndices: a.globalIndices,
				prompts,
				conf,
				batch,
				workerName: a.name || a.ws_url,
				onEvent
			});
		});
		return {
			promise: Promise.all(runs.map((r) => r.promise)),
			cancel: () => runs.forEach((r) => r.cancel()),
			workerCount: assignments.length
		};
	}

	// Legacy path: a single fileIds set assumed present on every worker (only
	// correct when distributing across one worker / one shared backend).
	const usable = workers.filter((w) => w.ws_url);
	if (!usable.length) throw new Error('No connectable workers');

	const shards = shardIndices(fileIds.length, usable.length);
	const runs = shards.map((indices, i) => {
		const worker = usable[i];
		onEvent?.({
			type: 'status',
			worker: worker.name || worker.ws_url,
			message: `Assigned ${indices.length} frame(s)`
		});
		return runSamOnWorker({
			wsUrl: worker.ws_url,
			samBaseUrl,
			fileIds: indices.map((idx) => fileIds[idx]),
			globalIndices: indices,
			prompts,
			conf,
			batch,
			workerName: worker.name || worker.ws_url,
			onEvent
		});
	});

	return {
		promise: Promise.all(runs.map((r) => r.promise)),
		cancel: () => runs.forEach((r) => r.cancel()),
		workerCount: usable.length
	};
}
