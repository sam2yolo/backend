<script>
	import { resolve } from '$app/paths';
	import { onDestroy, onMount } from 'svelte';
	import {
		createRoom,
		listWorkers,
		randomWorkerName,
		bootstrapCommand,
		notebookBootstrapCell
	} from '$lib/broker';
	import { renderResult, decodeMask } from '$lib/sammask';
	import { buildYoloDataset } from '$lib/dataset';
	import { maskToPolygons } from '$lib/contours';
	import { sampleVideoFrames } from '$lib/videoframes';
	import { distributeSam, shardIndices } from '$lib/samrun';

	const tabs = [
		'Overview',
		'Remotes',
		'Import',
		'Inference',
		'Dataset',
		'Train',
		'Export',
		'About'
	];
	const vehicleTypes = ['Bus', 'Sedan', 'Motorcycle', 'Truck', 'Bicycle', 'Van', 'Taxi', 'SUV'];
	const cocoVehicleClasses = {
		Bicycle: 1,
		Sedan: 2,
		Van: 2,
		Taxi: 2,
		SUV: 2,
		Motorcycle: 3,
		Bus: 5,
		Truck: 7
	};
	const datasetImportModes = ['upload', 'drive', 'direct'];
	const fleetSessionStorageKey = 'samtoyoloFleetSession';
	const imageExtensionPattern = /\.(jpe?g|png|webp|bmp)$/i;
	const videoExtensionPattern = /\.(mp4|mov|m4v|webm|avi|mkv|dav)$/i;
	const mediaExtensionPattern = /\.(jpe?g|png|webp|bmp|mp4|mov|m4v|webm|avi|mkv|dav)$/i;
	const datasetExtensionPattern =
		/\.(zip|ya?ml|txt|json|csv|jpe?g|png|webp|bmp|mp4|mov|m4v|webm|avi|mkv|dav)$/i;
	const localMediaAccept =
		'image/*,video/*,.jpg,.jpeg,.png,.webp,.bmp,.mp4,.mov,.m4v,.webm,.avi,.mkv,.dav';
	const datasetAccept =
		'.zip,.yaml,.yml,.txt,.json,.csv,image/*,video/*,.mp4,.mov,.m4v,.webm,.avi,.mkv,.dav';

	let activeTab = $state('Overview');
	let toast = $state('');
	let showMegaModal = $state(false);
	let showMenu = $state(false);
	let socket;
	let connectionState = $state('Connecting');
	let backendUrl = $state('');
	let remotes = $state([]);
	let modelState = $state('Not loaded');
	let modelReady = $state(false);
	let modelVariant = $state('yolo11n');
	let pendingTasks = []; // tasks to queue once the right model is ready
	let loadedModelType = $state(''); // 'sam' | 'yolo' — which model is currently loaded
	let requestedModelType = ''; // model type being initialized
	// Inference defaults to SAM 3.1 (text-prompted segmentation); YOLO is for training.
	let inferenceModel = $state('sam');
	let samBaseUrl = $state('http://127.0.0.1:8001');
	let newPromptText = $state('');
	let samBatch = $state(4);
	let samFps = $state(2); // frames/sec to sample from video for SAM
	let samMaxFrames = $state(40); // cap sampled frames (0 = all)
	// --- Fleet: auto-created room + workers discovered via the broker ---
	let fleetRoomId = $state('');
	let fleetRoomSecret = $state('');
	let fleetWorkerName = $state('');
	let fleetWorkers = $state([]); // [{ tunnel_id, name, remote_port, http_url, ws_url, alive, reachable, status }]
	let fleetBusy = $state(false);
	let fleetError = $state('');
	let fleetPollTimer = null;
	// tunnel_ids the user has explicitly excluded from runs (default: all included).
	let excludedWorkerIds = $state([]);

	// A worker is usable if reachable (fall back to alive for back-compat).
	const isWorkerReachable = (w) => (w.reachable === undefined ? !!w.alive : !!w.reachable);
	const reachableWorkers = $derived(fleetWorkers.filter(isWorkerReachable));
	// Selected = reachable AND not excluded; these fan out across a SAM run.
	const selectedWorkers = $derived(
		reachableWorkers.filter((w) => w.ws_url && !excludedWorkerIds.includes(w.tunnel_id))
	);

	function toggleWorker(worker) {
		const id = worker.tunnel_id;
		excludedWorkerIds = excludedWorkerIds.includes(id)
			? excludedWorkerIds.filter((x) => x !== id)
			: [...excludedWorkerIds, id];
	}

	let uploadedImports = $state([]);
	let selectedImportId = $state('');
	let selectedTrainingAnchorId = $state('');
	let importMode = $state('upload');
	let localFileInput = $state();
	let uploadBusy = $state(false);
	let googleDriveUrl = $state('');
	let uploadProgress = $state({
		visible: false,
		active: false,
		kind: '',
		fileId: '',
		label: '',
		detail: '',
		value: 0
	});
	let uploadProgressTimer;

	let promptTags = $state([]);
	let selectedPromptType = $state('');
	let savedPrompts = $state([]);
	let promptSaveName = $state('');
	let confidence = $state(0.25);
	let iou = $state(0.45);
	let inferenceImageSize = $state(640);
	let inferenceBatch = $state(16);
	let temporalDownsampling = $state(true);
	let frameKeepRate = $state(0.05);

	let datasetPath = $state('');
	let datasetMode = $state('upload');
	let datasetFileInput = $state();
	let datasetDirectUrl = $state('');
	let selectedDatasetId = $state('');
	let datasetManifestName = $state('traffic-dataset');
	let datasetRootPath = $state('');
	let mergeAssetIds = $state([]);
	let mappingSourceClass = $state('');
	let mappingTargetClass = $state('');
	let classMappings = $state([]);
	let savedDatasets = $state([]);
	let trainingName = $state('traffic-fast');
	let trainingEpochs = $state(50);
	let trainingBatch = $state(-1);
	let trainingWorkers = $state(8);
	let trainingImageSize = $state(640);
	let trainingDevice = $state('0');
	let trainingProject = $state('runs/train');

	let tasks = $state([]);

	// Realtime results: original image URLs (by backend file_id) + streamed frames.
	let originalUrls = $state({}); // file_id -> object URL (for overlay backgrounds)
	let liveResults = $state([]); // [{ key, taskId, className, frameIndex, result, imgUrl, detections }]
	let totalDetections = $state(0);
	let exportingDataset = $state(false);
	let datasetStats = $state(null);
	let exportFormat = $state('detect'); // 'detect' | 'segment'

	function defaultBackendUrl() {
		return `${location.protocol}//${location.hostname}:8000`;
	}

	function wsUrl() {
		return `${backendUrl.replace(/^http/, 'ws').replace(/\/$/, '')}/ws`;
	}

	function send(action, payload = {}) {
		if (!socket || socket.readyState !== WebSocket.OPEN) {
			notify('Backend is not connected');
			return false;
		}
		socket.send(JSON.stringify({ action, payload }));
		return true;
	}

	function clampProgress(value) {
		const number = Number(value);
		if (!Number.isFinite(number)) return 0;
		return Math.max(0, Math.min(100, Math.round(number)));
	}

	function formatBytes(bytes = 0) {
		const value = Number(bytes) || 0;
		if (value < 1024) return `${value} B`;
		const units = ['KB', 'MB', 'GB', 'TB'];
		let size = value / 1024;
		let unit = units[0];
		for (let i = 1; i < units.length && size >= 1024; i += 1) {
			size /= 1024;
			unit = units[i];
		}
		return `${size >= 10 ? size.toFixed(0) : size.toFixed(1)} ${unit}`;
	}

	function startUploadProgress(kind, label, detail = 'Preparing upload') {
		clearTimeout(uploadProgressTimer);
		uploadProgress = {
			visible: true,
			active: true,
			kind,
			fileId: '',
			label,
			detail,
			value: 0
		};
	}

	function updateUploadProgress(attrs = {}) {
		clearTimeout(uploadProgressTimer);
		uploadProgress = {
			...uploadProgress,
			visible: true,
			active: attrs.active ?? uploadProgress.active,
			...attrs,
			value: clampProgress(attrs.value ?? uploadProgress.value)
		};
	}

	function finishUploadProgress(detail = 'Completed') {
		updateUploadProgress({ active: false, detail, value: 100 });
		uploadProgressTimer = setTimeout(() => {
			uploadProgress = { ...uploadProgress, visible: false };
		}, 1800);
	}

	function failUploadProgress(detail = 'Upload failed') {
		updateUploadProgress({ active: false, detail });
		uploadProgressTimer = setTimeout(() => {
			uploadProgress = { ...uploadProgress, visible: false };
		}, 3200);
	}

	function remoteDownloadLabel(payload = {}) {
		const name = payload.file_name || payload.expected_path?.split('/').pop();
		if (name) return `Downloading ${name}`;
		if (uploadProgress.kind === 'direct-download') return 'Direct URL download';
		return 'Google Drive download';
	}

	function uploadFilesViaApi(items, options = {}) {
		const {
			names = [],
			targetUrl = backendUrl,
			label = 'Direct upload',
			detail = `${items.length} file${items.length === 1 ? '' : 's'} selected`
		} = options;

		startUploadProgress('local', label, detail);
		const body = new FormData();
		items.forEach((item, index) => {
			body.append('files', item, names[index] || item.name || `upload-${index + 1}`);
		});
		body.append('backendUrl', targetUrl);

		return new Promise((resolve, reject) => {
			const xhr = new XMLHttpRequest();
			xhr.open('POST', '/api/uploads');
			xhr.responseType = 'json';

			xhr.upload.addEventListener('progress', (event) => {
				if (!event.lengthComputable) {
					updateUploadProgress({
						value: Math.max(uploadProgress.value, 5),
						detail: 'Uploading to website'
					});
					return;
				}

				const value = Math.min(99, (event.loaded / event.total) * 100);
				updateUploadProgress({
					value,
					detail:
						value >= 99
							? 'Saving to backend'
							: `${formatBytes(event.loaded)} of ${formatBytes(event.total)}`
				});
			});

			xhr.upload.addEventListener('load', () => {
				updateUploadProgress({ value: 99, detail: 'Saving to backend' });
			});

			xhr.addEventListener('load', () => {
				let result = xhr.response;
				if (!result && xhr.responseText) {
					try {
						result = JSON.parse(xhr.responseText);
					} catch {
						result = {};
					}
				}
				if (xhr.status >= 200 && xhr.status < 300) {
					resolve(result || {});
					return;
				}
				reject(new Error(result?.message || result?.error || 'Upload failed'));
			});
			xhr.addEventListener('error', () => reject(new Error('Upload failed')));
			xhr.addEventListener('abort', () => reject(new Error('Upload cancelled')));
			xhr.send(body);
		});
	}

	function connectBackend() {
		socket?.close?.();
		connectionState = 'Connecting';
		modelReady = false;
		modelState = 'Not loaded';

		try {
			socket = new WebSocket(wsUrl());
			socket.addEventListener('open', () => {
				connectionState = 'Connected';
				send('ping');
				send('list_files');
				send('list_models');
			});
			socket.addEventListener('message', handleBackendMessage);
			socket.addEventListener('error', () => {
				connectionState = 'Connection error';
			});
			socket.addEventListener('close', () => {
				connectionState = 'Disconnected';
			});
		} catch {
			connectionState = 'Invalid backend URL';
		}
	}

	// --- Fleet / room management ------------------------------------------
	function saveFleetSession() {
		if (!fleetRoomId || !fleetRoomSecret) return;
		localStorage.setItem(
			fleetSessionStorageKey,
			JSON.stringify({
				room_id: fleetRoomId,
				room_secret: fleetRoomSecret,
				worker_name: fleetWorkerName
			})
		);
	}

	function restoreFleetSession() {
		try {
			const raw = localStorage.getItem(fleetSessionStorageKey);
			if (!raw) return false;
			const session = JSON.parse(raw);
			if (!session?.room_id || !session?.room_secret) return false;
			fleetRoomId = session.room_id;
			fleetRoomSecret = session.room_secret;
			fleetWorkerName = session.worker_name || randomWorkerName();
			return true;
		} catch {
			localStorage.removeItem(fleetSessionStorageKey);
			return false;
		}
	}

	async function createFleetRoom({ auto = false } = {}) {
		if (fleetBusy || fleetRoomId) return;
		fleetBusy = true;
		fleetError = '';
		try {
			const room = await createRoom('fleet-vision');
			fleetRoomId = room.room_id;
			fleetRoomSecret = room.room_secret;
			if (!fleetWorkerName) fleetWorkerName = randomWorkerName();
			saveFleetSession();
			notify(auto ? 'Notebook room is ready' : 'Room created');
			startFleetPolling();
		} catch (error) {
			fleetError = error?.message || 'Could not create room';
			notify(fleetError);
		} finally {
			fleetBusy = false;
		}
	}

	function currentFleetWorkerName() {
		if (!fleetWorkerName) {
			fleetWorkerName = randomWorkerName();
			saveFleetSession();
		}
		return fleetWorkerName;
	}

	function fleetCommand() {
		if (!fleetRoomId) return '';
		return bootstrapCommand(fleetRoomId, fleetRoomSecret, currentFleetWorkerName());
	}

	function fleetNotebookCell() {
		if (!fleetRoomId || !fleetRoomSecret) return '';
		return notebookBootstrapCell(fleetRoomId, fleetRoomSecret, currentFleetWorkerName());
	}

	function regenerateWorkerName() {
		fleetWorkerName = randomWorkerName();
		saveFleetSession();
	}

	async function refreshFleetWorkers() {
		if (!fleetRoomId) return;
		try {
			fleetWorkers = await listWorkers(fleetRoomId);
		} catch (error) {
			fleetError = error?.message || 'Could not list workers';
		}
	}

	function startFleetPolling() {
		stopFleetPolling();
		refreshFleetWorkers();
		fleetPollTimer = setInterval(refreshFleetWorkers, 4000);
	}

	function stopFleetPolling() {
		if (fleetPollTimer) {
			clearInterval(fleetPollTimer);
			fleetPollTimer = null;
		}
	}

	function copyFleetCommand() {
		const command = fleetCommand();
		if (!command) return;
		navigator.clipboard?.writeText(command);
		notify('Terminal command copied');
	}

	function copyFleetNotebookCell() {
		const cell = fleetNotebookCell();
		if (!cell) return;
		navigator.clipboard?.writeText(cell);
		notify('Notebook cell copied');
	}

	function connectToWorker(worker) {
		backendUrl = worker.http_url;
		connectBackend();
		setTab('Inference');
		notify(`Connecting to ${worker.name}`);
	}

	function upsertTask(id, attrs = {}) {
		if (!id) return;
		const index = tasks.findIndex((task) => task.id === id);
		if (index === -1) {
			tasks = [
				...tasks,
				{
					id,
					name: attrs.name || id,
					status: attrs.status || 'Pending',
					progress: attrs.progress ?? 0,
					chunks: [],
					...attrs
				}
			];
		} else {
			tasks = tasks.map((task, taskIndex) => (taskIndex === index ? { ...task, ...attrs } : task));
		}
	}

	function errorMessage(payload) {
		return payload.error || payload.message || 'Backend operation failed';
	}

	function handleBackendMessage(event) {
		try {
			const { action, payload = {} } = JSON.parse(event.data);
			const id = payload.task_id || payload.id;

			switch (action) {
				case 'pong':
					connectionState = 'Connected';
					break;
				case 'file_list':
					uploadedImports = (payload.files || []).map(normalizeBackendFile);
					reconcileAssetSelections();
					break;
				case 'list_models_response':
					remotes = payload.models || [];
					break;
				case 'model_setup_started':
					modelState = 'Preparing environment';
					break;
				case 'model_setup_completed':
					modelState = 'Loading weights';
					break;
				case 'model_init_started':
					modelState = `Loading ${payload.model_name || modelVariant}`;
					break;
				case 'model_init_completed':
					modelReady = true;
					loadedModelType = requestedModelType || loadedModelType;
					modelState = payload.model_name || loadedModelType || modelVariant;
					notify(`${modelState} is ready`);
					queuePendingTasks();
					break;
				case 'model_setup_error':
				case 'model_init_error':
					modelReady = false;
					modelState = 'Failed';
					pendingTasks = [];
					notify(errorMessage(payload));
					break;
				case 'file_download_initiated':
					upsertTask(payload.file_id, {
						name: payload.file_name || 'Google Drive import',
						status: 'Downloading',
						progress: 0
					});
					updateUploadProgress({
						kind: uploadProgress.kind || 'drive',
						fileId: payload.file_id,
						label: remoteDownloadLabel(payload),
						detail: 'Download started',
						value: 0
					});
					break;
				case 'download_progress':
					upsertTask(payload.file_id, { status: 'Downloading', progress: payload.progress ?? 0 });
					updateUploadProgress({
						kind: uploadProgress.kind || 'drive',
						fileId: payload.file_id || uploadProgress.fileId,
						label: uploadProgress.label || remoteDownloadLabel(payload),
						detail:
							payload.progress == null
								? 'Downloading'
								: `${clampProgress(payload.progress)}%${
										payload.total_size ? ` of ${payload.total_size}` : ''
									}`,
						value: payload.progress == null ? Math.max(uploadProgress.value, 5) : payload.progress
					});
					break;
				case 'file_download_completed':
					upsertTask(payload.file_id, { status: 'Completed', progress: 100 });
					googleDriveUrl = '';
					datasetDirectUrl = '';
					uploadBusy = false;
					finishUploadProgress('Download complete');
					send('list_files');
					notify('Backend download completed');
					break;
				case 'download_failed':
					upsertTask(payload.file_id, { status: 'Failed' });
					uploadBusy = false;
					failUploadProgress(errorMessage(payload));
					notify(errorMessage(payload));
					break;
				case 'task_added':
					upsertTask(payload.id, {
						name:
							payload.name ||
							(payload.type === 'train' ? 'YOLO training' : selectedImport()?.name) ||
							payload.id,
						status: 'Queued',
						progress: 0,
						type: payload.type || 'inference',
						// remember enough to map chunk results back to source frames
						file_ids: payload.file_ids || (payload.file_id ? [payload.file_id] : []),
						batch: payload.batch || 1,
						className: payload.class_name || payload.text_prompt || null
					});
					send('start_inference_from_queue');
					break;
				case 'work_started':
					notify('Backend worker started');
					break;
				case 'inference_stage_plus_progress':
					upsertTask(id, {
						status: payload.stage || 'Inferencing',
						progress: payload.progress ?? 0
					});
					break;
				case 'inference_task_chunk_result': {
					const current = tasks.find((task) => task.id === payload.task_id);
					const chunks = [...(current?.chunks || []), payload.chunk_id];
					upsertTask(payload.task_id, { chunks });
					// Pull the chunk's JSON so we can render masks live (SAM only —
					// SAM chunk data is plain JSON; YOLO chunks are pickled objects).
					if (current?.className != null || loadedModelType === 'sam') {
						send('fetch_inference_chunk', {
							task_id: payload.task_id,
							chunk_id: payload.chunk_id
						});
					}
					break;
				}
				case 'inference_chunk_data':
					ingestChunkData(payload);
					break;
				case 'inference_completed':
					upsertTask(id, { status: 'Completed', progress: 100 });
					notify('Traffic inference completed');
					break;
				case 'training_started':
					upsertTask(id, { status: `Training · ${payload.epochs} epochs`, progress: 5 });
					break;
				case 'training_completed':
					upsertTask(id, { status: 'Completed', progress: 100, resultPath: payload.result_path });
					notify(`Training completed: ${payload.result_path}`);
					break;
				case 'task_failed':
				case 'inference_task_error':
				case 'create_inference_task_error':
				case 'create_infrerence_task_error':
					upsertTask(id || 'backend-error', { status: 'Failed', progress: 0 });
					notify(errorMessage(payload));
					break;
				case 'task_cancelled':
					upsertTask(id, { status: 'Cancelled' });
					break;
				case 'delete_file_success':
					send('list_files');
					break;
				case 'delete_file_failed':
					notify(errorMessage(payload));
					break;
			}
		} catch {
			notify('Backend sent an unreadable response');
		}
	}

	function selectedImport() {
		return mediaAssets().find((file) => file.id === selectedImportId);
	}

	function normalizeBackendFile(file) {
		return {
			...file,
			id: file.id,
			originalName: file.originalName || file.original_name || '',
			converted: Boolean(file.converted),
			conversionError: file.conversionError || file.conversion_error || ''
		};
	}

	function hasAssetId(list, id) {
		return !!id && list.some((file) => file.id === id);
	}

	function reconcileAssetSelections({
		preferredMediaId = '',
		preferredDatasetId = '',
		preferredTrainingAnchorId = preferredMediaId
	} = {}) {
		const media = mediaAssets();
		const datasets = datasetAssets();

		if (hasAssetId(media, preferredMediaId)) selectedImportId = preferredMediaId;
		else if (!hasAssetId(media, selectedImportId)) selectedImportId = media[0]?.id || '';

		if (hasAssetId(media, preferredTrainingAnchorId))
			selectedTrainingAnchorId = preferredTrainingAnchorId;
		else if (!hasAssetId(media, selectedTrainingAnchorId))
			selectedTrainingAnchorId = media[0]?.id || '';

		if (hasAssetId(datasets, preferredDatasetId)) selectedDatasetId = preferredDatasetId;
		else if (!hasAssetId(datasets, selectedDatasetId)) selectedDatasetId = datasets[0]?.id || '';
	}

	function isMediaFile(file) {
		return mediaExtensionPattern.test(file?.name || '');
	}

	function mediaAssets() {
		return uploadedImports.filter(isMediaFile);
	}

	function selectedTrainingAnchor() {
		return mediaAssets().find((file) => file.id === selectedTrainingAnchorId);
	}

	function detectFileType(name = '') {
		return imageExtensionPattern.test(name) ? 'image' : 'video';
	}

	function fileKind(file) {
		const name = file?.name || '';
		if (/\.(zip)$/i.test(name)) return 'dataset zip';
		if (/\.(ya?ml)$/i.test(name)) return 'data config';
		if (/\.(txt|json|csv)$/i.test(name)) return 'labels';
		if (imageExtensionPattern.test(name)) return 'image';
		if (videoExtensionPattern.test(name)) return 'video';
		return 'asset';
	}

	function isDavFile(file) {
		return /\.dav$/i.test(file?.name || '') || /\.dav$/i.test(file?.path || '');
	}

	function datasetAssets() {
		return uploadedImports.filter((file) => datasetExtensionPattern.test(file.name || ''));
	}

	function selectedDatasetAsset() {
		return datasetAssets().find((file) => file.id === selectedDatasetId);
	}

	function resultDownloadUrl(chunkId) {
		const params = new URLSearchParams({ id: chunkId, backend: backendUrl });
		return `${resolve('/api/results')}?${params.toString()}`;
	}

	function backendFileUrl(file) {
		const params = new URLSearchParams({ id: file.id, backend: backendUrl });
		return `${resolve('/api/files')}?${params.toString()}`;
	}

	async function ensureOriginalUrl(file) {
		if (originalUrls[file.id]) return originalUrls[file.id];
		const response = await fetch(backendFileUrl(file));
		if (!response.ok) {
			const message = await response.text();
			throw new Error(message || 'Could not load backend video for sampling');
		}
		const blob = await response.blob();
		const url = URL.createObjectURL(blob);
		originalUrls = { ...originalUrls, [file.id]: url };
		return url;
	}

	async function downloadResult(chunkId) {
		try {
			const response = await fetch(resultDownloadUrl(chunkId));
			if (!response.ok) throw new Error('Result download failed');
			const blob = await response.blob();
			const url = URL.createObjectURL(blob);
			const link = document.createElement('a');
			link.href = url;
			link.download = `inference-${chunkId}.pkl`;
			link.click();
			URL.revokeObjectURL(url);
		} catch (error) {
			notify(error instanceof Error ? error.message : 'Result download failed');
		}
	}

	function ensureModel(modelType, taskList) {
		if (connectionState !== 'Connected') {
			notify('Connect to the backend first');
			return;
		}
		pendingTasks = Array.isArray(taskList) ? taskList : [taskList];
		// Reuse the loaded model only if it's the same type and ready.
		if (modelReady && loadedModelType === modelType) {
			queuePendingTasks();
			return;
		}
		modelReady = false;
		requestedModelType = modelType;
		modelState = 'Starting';
		if (modelType === 'sam') {
			send('init_model', { model_name: 'sam', base_url: samBaseUrl });
			notify('Initializing SAM 3.1 on the worker');
		} else {
			send('init_model', { model_name: 'yolo', variant_name: modelVariant });
			notify('Preparing the YOLO environment; the first run can take longer');
		}
	}

	function addPrompt() {
		const value = newPromptText.trim();
		if (value && !promptTags.includes(value)) {
			promptTags = [...promptTags, value];
		}
		newPromptText = '';
	}

	function initializeModel() {
		if (modelReady) {
			notify(`${modelState} is already ready`);
			return;
		}
		if (connectionState !== 'Connected') {
			notify('Connect to the backend first');
			return;
		}
		modelState = 'Starting';
		requestedModelType = 'yolo';
		send('init_model', { model_name: 'yolo', variant_name: modelVariant });
	}

	function queuePendingTasks() {
		if (!pendingTasks.length) return;
		const queue = pendingTasks;
		pendingTasks = [];
		for (const task of queue) send('create_inference_task', task);
	}

	function startInference() {
		const file = selectedImport();
		if (!file) {
			notify('Select a backend import first');
			return;
		}

		if (inferenceModel === 'sam') {
			startSamInference(file);
			return;
		}

		const classes = [
			...new Set(promptTags.map((tag) => cocoVehicleClasses[tag]).filter(Number.isInteger))
		];
		ensureModel('yolo', {
			name: `Traffic · ${file.name}`,
			file_id: file.id,
			file_ids: [file.id],
			file_type: detectFileType(file.name),
			conf: Number(confidence),
			iou: Number(iou),
			imgsz: Number(inferenceImageSize),
			batch: Number(inferenceBatch),
			classes: classes.length ? classes : [1, 2, 3, 5, 7],
			temporal_downsampling: temporalDownsampling,
			drop_rate: Number(frameKeepRate)
		});
	}

	// Upload an array of Blobs (e.g. sampled video frames) as images; returns the
	// backend file records (in order).
	async function uploadBlobs(blobs, names, targetUrl = backendUrl, label = 'Frame upload') {
		const result = await uploadFilesViaApi(blobs, {
			names,
			targetUrl,
			label,
			detail: `${blobs.length} sampled frame${blobs.length === 1 ? '' : 's'}`
		});
		if (!Array.isArray(result.files)) throw new Error(result.message || 'Frame upload failed');
		finishUploadProgress(
			`${result.files.length} frame${result.files.length === 1 ? '' : 's'} saved`
		);
		return result.files;
	}

	// SAM 3.1: image-only, one task per text prompt. Each prompt becomes a class
	// in the compiled dataset (prompt index = class id). Videos are sampled into
	// frames in the browser first.
	async function startSamInference(file) {
		if (!promptTags.length) {
			notify('Add at least one text prompt for SAM');
			return;
		}

		let fileIds;
		if (detectFileType(file.name) === 'image') {
			fileIds = [file.id];
		} else {
			if (isDavFile(file)) {
				notify('DAV files can be selected for YOLO; convert to MP4 before using SAM sampling');
				return;
			}
			try {
				if (!originalUrls[file.id]) {
					modelState = 'Loading source video';
					notify('Loading source video from backend');
				}
				const videoUrl = await ensureOriginalUrl(file);
				notify(`Sampling video at ${samFps} fps…`);
				const frames = await sampleVideoFrames(
					videoUrl,
					Number(samFps),
					Number(samMaxFrames) || 0,
					(done, total) => (modelState = `Sampling ${done}/${total}`)
				);
				if (!frames.length) {
					notify('No frames sampled from video');
					return;
				}
				const blobs = frames.map((f) => f.blob);
				const names = frames.map((_, i) => `${file.name}-f${String(i).padStart(5, '0')}.jpg`);

				// Files live on the worker they're uploaded to (each worker is its own
				// backend), so shard frame indices first and upload each shard to its
				// own worker; build per-worker assignments for distribution.
				const pool = selectedWorkers
					.filter((w) => w.http_url && w.ws_url)
					.map((w) => ({ name: w.name, http_url: w.http_url, ws_url: w.ws_url }));

				if (pool.length) {
					const shards = shardIndices(frames.length, pool.length);
					notify(`Uploading ${frames.length} frames across ${shards.length} worker(s)…`);
					const assignments = [];
					const urls = {};
					for (let i = 0; i < shards.length; i++) {
						const indices = shards[i];
						const w = pool[i];
						const shardBlobs = indices.map((idx) => blobs[idx]);
						const shardNames = indices.map((idx) => names[idx]);
						modelState = `Uploading shard ${i + 1}/${shards.length} to ${w.name}`;
						const uploaded = await uploadBlobs(
							shardBlobs,
							shardNames,
							w.http_url,
							`Frame upload ${i + 1}/${shards.length}`
						);
						const ids = uploaded.map((u) => u.id);
						ids.forEach((id, k) => (urls[id] = URL.createObjectURL(shardBlobs[k])));
						assignments.push({
							name: w.name,
							ws_url: w.ws_url,
							fileIds: ids,
							globalIndices: indices
						});
					}
					originalUrls = { ...originalUrls, ...urls };
					liveResults = [];
					totalDetections = 0;
					runDistributedSam(
						assignments.map((a) => ({ name: a.name, ws_url: a.ws_url })),
						null,
						{ assignments, totalFrames: frames.length }
					);
					return;
				}

				// No fleet workers selected — upload everything to the single backend.
				notify(`Uploading ${frames.length} sampled frames…`);
				const uploaded = await uploadBlobs(blobs, names, backendUrl, 'Sampled frame upload');
				fileIds = uploaded.map((u) => u.id);
				const urls = {};
				uploaded.forEach((u, i) => (urls[u.id] = URL.createObjectURL(blobs[i])));
				originalUrls = { ...originalUrls, ...urls };
			} catch (error) {
				if (uploadProgress.active) failUploadProgress(error?.message || 'Frame upload failed');
				notify(error?.message || 'Video sampling failed');
				return;
			}
		}

		// Single-backend run: a still image, or a video with no fleet workers
		// selected. The file already lives on backendUrl, so run it there.
		const backendWorker = {
			name: backendUrl,
			ws_url: backendUrl.replace(/^http/, 'ws').replace(/\/$/, '') + '/ws'
		};
		liveResults = [];
		totalDetections = 0;
		runDistributedSam([backendWorker], fileIds);
	}

	// Orchestrate a distributed SAM run across workers and stream results into the
	// live view + a task-project row. The global socket is freed during the run
	// (each worker allows a single WebSocket) and reconnected afterwards.
	function runDistributedSam(workers, fileIds, extra = {}) {
		const { assignments = null, totalFrames = null } = extra;
		const prompts = promptTags.slice();
		const frameCount = assignments
			? (totalFrames ?? assignments.reduce((n, a) => n + a.fileIds.length, 0))
			: fileIds.length;
		const workerCount = assignments ? assignments.length : workers.length;
		const totalUnits = frameCount * prompts.length;
		let doneUnits = 0;
		const jobId = 'sam-' + Date.now().toString(36);
		upsertTask(jobId, {
			name: `SAM · ${prompts.join(', ')}`,
			status: `Running on ${workerCount} worker${workerCount === 1 ? '' : 's'}`,
			progress: 0,
			type: 'inference'
		});

		const hadSocket = socket && socket.readyState === WebSocket.OPEN;
		socket?.close?.();
		connectionState = 'Running SAM';

		let job;
		try {
			job = distributeSam(workers, {
				fileIds,
				assignments,
				prompts,
				samBaseUrl,
				conf: Number(confidence),
				batch: Number(samBatch),
				onEvent: (e) => {
					if (e.type === 'status') {
						modelState = `${e.worker}: ${e.message}`;
					} else if (e.type === 'frame') {
						const detections = (e.result.masks || []).length;
						totalDetections += detections;
						doneUnits += 1;
						upsertTask(jobId, {
							progress: Math.round((doneUnits / Math.max(1, totalUnits)) * 100),
							status: `Running · ${doneUnits}/${totalUnits}`
						});
						liveResults = [
							{
								key: `${e.taskId}-${e.frameIndex}-${e.className}`,
								taskId: e.taskId,
								className: e.className || 'objects',
								frameIndex: e.frameIndex,
								result: e.result,
								imgUrl: e.fileId ? originalUrls[e.fileId] : null,
								detections
							},
							...liveResults
						].slice(0, 300);
					} else if (e.type === 'error') {
						notify(`${e.worker}: ${e.message}`);
					}
				}
			});
		} catch (error) {
			upsertTask(jobId, { status: 'Failed' });
			notify(error?.message || 'Could not start SAM run');
			if (hadSocket) connectBackend();
			return;
		}

		job.promise
			.then(() => {
				upsertTask(jobId, { status: 'Completed', progress: 100 });
				modelState = 'SAM complete';
				notify('SAM inference complete');
			})
			.catch((error) => {
				upsertTask(jobId, { status: 'Failed' });
				notify(error?.message || 'SAM run failed');
			})
			.finally(() => {
				if (hadSocket) connectBackend();
			});
	}

	// Turn a fetched SAM chunk into renderable live-result tiles.
	function ingestChunkData(payload) {
		const task = tasks.find((t) => t.id === payload.task_id);
		const images = payload.data?.images || [];
		const batch = task?.batch || 1;
		const fileIds = task?.file_ids || [];
		const className = task?.className || 'objects';
		const chunkIndex = payload.chunk_index ?? 0;
		const additions = images.map((result, j) => {
			const globalIdx = chunkIndex * batch + j;
			const fileId = fileIds[globalIdx];
			const detections = (result.masks || []).length;
			totalDetections += detections;
			return {
				key: `${payload.task_id}-${payload.chunk_id}-${j}`,
				taskId: payload.task_id,
				className,
				frameIndex: globalIdx,
				result,
				imgUrl: fileId ? originalUrls[fileId] : null,
				detections
			};
		});
		liveResults = [...additions, ...liveResults].slice(0, 200);
	}

	// Svelte action: paint a SAM result onto a <canvas>, over its source image.
	function drawResult(node, params) {
		function paint(p) {
			if (!p?.result) return;
			if (p.imgUrl) {
				const img = new Image();
				img.onload = () => renderResult(node, img, p.result);
				img.onerror = () => renderResult(node, null, p.result);
				img.src = p.imgUrl;
			} else {
				renderResult(node, null, p.result);
			}
		}
		paint(params);
		return { update: paint };
	}

	// Compile the streamed SAM results into a YOLO-compatible dataset zip.
	// Each prompt is a class; boxes become YOLO detection labels.
	async function exportYoloDataset() {
		if (!liveResults.length) {
			notify('Run SAM inference first to generate results');
			return;
		}
		const classes = promptTags.slice();
		if (!classes.length) {
			notify('No prompt classes available to label');
			return;
		}
		exportingDataset = true;
		const segment = exportFormat === 'segment';
		try {
			// Group all per-prompt results by source frame.
			const byFrame = new Map();
			for (const item of liveResults) {
				const f = item.frameIndex;
				if (!byFrame.has(f)) {
					byFrame.set(f, {
						name: `frame_${String(f).padStart(6, '0')}`,
						imgUrl: item.imgUrl,
						width: item.result.width,
						height: item.result.height,
						boxes: [],
						polygons: []
					});
				}
				const entry = byFrame.get(f);
				const classId = classes.indexOf(item.className);
				if (classId < 0) continue;
				const boxes = item.result.boxes || [];
				for (const b of boxes) {
					entry.boxes.push({ classId, x1: b[0], y1: b[1], x2: b[2], y2: b[3] });
				}
				if (segment) {
					const masks = item.result.masks || [];
					if (masks.length) {
						masks.forEach((maskObj, mi) => {
							let decoded;
							try {
								decoded = decodeMask(maskObj);
							} catch {
								decoded = null;
							}
							if (!decoded) return;
							const polys = maskToPolygons(decoded.bits, decoded.w, decoded.h, {
								minArea: 16,
								simplifyTol: 1.5
							});
							if (polys.length) {
								for (const p of polys) entry.polygons.push({ classId, points: p.points });
							} else if (boxes[mi]) {
								// Tracer found nothing usable — fall back to the box rectangle.
								const [x1, y1, x2, y2] = boxes[mi];
								entry.polygons.push({
									classId,
									points: [
										[x1, y1],
										[x2, y1],
										[x2, y2],
										[x1, y2]
									]
								});
							}
						});
					} else {
						// No masks — fall back to each box as a 4-point polygon.
						for (const b of boxes) {
							entry.polygons.push({
								classId,
								points: [
									[b[0], b[1]],
									[b[2], b[1]],
									[b[2], b[3]],
									[b[0], b[3]]
								]
							});
						}
					}
				}
			}

			const frames = [];
			for (const entry of byFrame.values()) {
				let imageBlob = null;
				if (entry.imgUrl) {
					try {
						imageBlob = await (await fetch(entry.imgUrl)).blob();
					} catch {
						imageBlob = null;
					}
				}
				frames.push({
					name: entry.name,
					imageBlob,
					width: entry.width,
					height: entry.height,
					boxes: entry.boxes,
					polygons: entry.polygons
				});
			}

			const datasetName = datasetManifestName?.trim() || 'sam-dataset';
			const { blob, frameCount, labeledFrames, boxesTotal } = await buildYoloDataset({
				classes,
				frames,
				name: datasetName,
				format: exportFormat
			});

			const link = document.createElement('a');
			link.href = URL.createObjectURL(blob);
			link.download = `${datasetName}.zip`;
			link.click();
			URL.revokeObjectURL(link.href);

			datasetStats = {
				frameCount,
				labeledFrames,
				boxesTotal,
				classes: classes.length,
				format: exportFormat
			};
			const unit = segment ? 'polygons' : 'boxes';
			notify(
				`Dataset exported: ${frameCount} frames, ${boxesTotal} ${unit}, ${classes.length} classes`
			);
		} catch (error) {
			notify(error?.message || 'Dataset export failed');
		} finally {
			exportingDataset = false;
		}
	}

	function startTraining() {
		const validationFile = selectedTrainingAnchor();
		const trainDataPath = datasetPath.trim() || datasetRootPath.trim();
		if (!validationFile) {
			notify('Select one uploaded image/video as the backend training anchor');
			return;
		}
		if (!trainDataPath) {
			notify('Enter a backend-visible data.yaml path or dataset zip path');
			return;
		}
		ensureModel('yolo', {
			type: 'train',
			name: trainingName.trim() || 'traffic-fast',
			file_id: validationFile.id,
			file_type: detectFileType(validationFile.name),
			dataset: trainDataPath,
			epochs: Number(trainingEpochs),
			batch_size: Number(trainingBatch),
			imgsz: Number(trainingImageSize),
			workers: Number(trainingWorkers),
			device: trainingDevice.trim() || '0',
			project: trainingProject.trim() || 'runs/train'
		});
	}

	onMount(() => {
		backendUrl = localStorage.getItem('backendUrl') || defaultBackendUrl();
		try {
			const raw = localStorage.getItem('savedPrompts');
			if (raw) savedPrompts = JSON.parse(raw);
		} catch {
			savedPrompts = [];
		}
		try {
			const rawDatasets = localStorage.getItem('savedDatasets');
			if (rawDatasets) savedDatasets = JSON.parse(rawDatasets);
		} catch {
			savedDatasets = [];
		}
		if (restoreFleetSession()) {
			saveFleetSession();
			startFleetPolling();
		} else {
			createFleetRoom({ auto: true });
		}
		connectBackend();
	});

	onDestroy(() => {
		socket?.close?.();
		stopFleetPolling();
		clearTimeout(uploadProgressTimer);
	});

	function saveBackendUrl() {
		backendUrl = backendUrl.trim().replace(/\/$/, '');
		localStorage.setItem('backendUrl', backendUrl);
		connectBackend();
	}

	function savePrompt() {
		const name = promptSaveName.trim();
		if (!name || savedPrompts.some((prompt) => prompt.name === name)) {
			notify(name ? 'Name already exists' : 'Enter a name to save prompt');
			return;
		}
		savedPrompts = [...savedPrompts, { name, tags: [...promptTags] }];
		localStorage.setItem('savedPrompts', JSON.stringify(savedPrompts));
		promptSaveName = '';
		notify('Prompt saved');
	}

	function setTab(tab) {
		activeTab = tab;
		showMenu = false;
	}

	function notify(message) {
		toast = String(message);
		setTimeout(() => {
			if (toast === String(message)) toast = '';
		}, 3200);
	}

	function copyCode() {
		navigator.clipboard?.writeText(backendUrl);
		notify('Backend URL copied');
	}

	function chooseLocalFiles() {
		localFileInput?.click();
	}

	function chooseDatasetFiles() {
		datasetFileInput?.click();
	}

	async function handleLocalFileChange(event) {
		const files = Array.from(event.currentTarget.files ?? []).filter(
			(file) =>
				file.type.startsWith('video/') ||
				file.type.startsWith('image/') ||
				mediaExtensionPattern.test(file.name)
		);
		event.currentTarget.value = '';
		if (!files.length) {
			notify('Choose an image or video file');
			return;
		}

		uploadBusy = true;
		try {
			const result = await uploadFilesViaApi(files, {
				label: 'Direct upload',
				detail: `${files.length} local file${files.length === 1 ? '' : 's'}`
			});
			if (!Array.isArray(result.files)) throw new Error(result.message || 'Upload failed');
			const records = result.files.map(normalizeBackendFile);
			// Keep local object URLs so we can overlay SAM masks on images and
			// sample frames from videos locally. result.files is returned in the
			// same order we appended `files`.
			const urls = {};
			records.forEach((record, index) => {
				const source = files[index];
				if (
					source &&
					!record.converted &&
					(source.type.startsWith('image/') ||
						source.type.startsWith('video/') ||
						mediaExtensionPattern.test(source.name))
				) {
					urls[record.id] = URL.createObjectURL(source);
				}
			});
			if (Object.keys(urls).length) originalUrls = { ...originalUrls, ...urls };
			uploadedImports = [...records, ...uploadedImports];
			reconcileAssetSelections({ preferredMediaId: records.find(isMediaFile)?.id || '' });
			notify(`${records.length} file${records.length === 1 ? '' : 's'} uploaded to backend`);
			finishUploadProgress(`${records.length} file${records.length === 1 ? '' : 's'} uploaded`);
		} catch (error) {
			failUploadProgress(error instanceof Error ? error.message : 'Upload failed');
			notify(error instanceof Error ? error.message : 'Upload failed');
		} finally {
			uploadBusy = false;
		}
	}

	async function uploadFilesToBackend(files, successLabel = 'file') {
		if (!files.length) {
			notify(`Choose at least one ${successLabel}`);
			return [];
		}

		uploadBusy = true;
		try {
			const result = await uploadFilesViaApi(files, {
				label: `${successLabel[0]?.toUpperCase() || 'F'}${successLabel.slice(1)} upload`,
				detail: `${files.length} ${successLabel}${files.length === 1 ? '' : 's'} selected`
			});
			if (!Array.isArray(result.files)) throw new Error(result.message || 'Upload failed');
			const records = result.files.map(normalizeBackendFile);
			uploadedImports = [...records, ...uploadedImports];
			reconcileAssetSelections({
				preferredMediaId: records.find(isMediaFile)?.id || '',
				preferredDatasetId: records[0]?.id || ''
			});
			notify(`${records.length} ${successLabel}${records.length === 1 ? '' : 's'} uploaded`);
			finishUploadProgress(
				`${records.length} ${successLabel}${records.length === 1 ? '' : 's'} uploaded`
			);
			return records;
		} catch (error) {
			failUploadProgress(error instanceof Error ? error.message : 'Upload failed');
			notify(error instanceof Error ? error.message : 'Upload failed');
			return [];
		} finally {
			uploadBusy = false;
		}
	}

	async function handleDatasetFileChange(event) {
		const files = Array.from(event.currentTarget.files ?? []);
		event.currentTarget.value = '';
		const uploaded = await uploadFilesToBackend(files, 'dataset asset');
		if (uploaded[0]) {
			selectedDatasetId = uploaded[0].id;
			if (/\.(ya?ml)$/i.test(uploaded[0].name)) datasetRootPath = uploaded[0].path;
		}
	}

	function uploadFromGoogleDrive() {
		if (!googleDriveUrl.trim()) {
			notify('Paste a Google Drive link');
			return;
		}
		if (send('download_file_google_drive', { url: googleDriveUrl.trim() })) {
			uploadBusy = true;
			startUploadProgress('drive', 'Google Drive download', 'Waiting for backend');
			notify('Backend download started');
		}
	}

	function downloadDatasetFromGoogleDrive() {
		if (!googleDriveUrl.trim()) {
			notify('Paste a Google Drive dataset link');
			return;
		}
		if (send('download_file_google_drive', { url: googleDriveUrl.trim() })) {
			uploadBusy = true;
			startUploadProgress('drive', 'Google Drive dataset download', 'Waiting for backend');
			notify('Backend dataset download started');
		}
	}

	function downloadDatasetDirectUrl() {
		if (!datasetDirectUrl.trim()) {
			notify('Paste a direct dataset URL');
			return;
		}
		if (send('download_file_wget', { url: datasetDirectUrl.trim() })) {
			uploadBusy = true;
			startUploadProgress('direct-download', 'Direct URL download', 'Waiting for backend');
			notify('Backend direct dataset download started');
		}
	}

	function removeImport(file) {
		send('delete_file', { file_id: file.id });
		uploadedImports = uploadedImports.filter((item) => item.id !== file.id);
		if (selectedImportId === file.id) selectedImportId = mediaAssets()[0]?.id || '';
		if (selectedDatasetId === file.id) selectedDatasetId = datasetAssets()[0]?.id || '';
		if (selectedTrainingAnchorId === file.id) selectedTrainingAnchorId = mediaAssets()[0]?.id || '';
		mergeAssetIds = mergeAssetIds.filter((id) => id !== file.id);
	}

	function useDatasetAsset(file) {
		if (!file) {
			notify('Select a dataset asset first');
			return;
		}
		selectedDatasetId = file.id;
		datasetRootPath = file.path;
		datasetPath = file.path;
		if (/\.(zip)$/i.test(file.name || '')) {
			notify('Zip selected. If training fails, enter the backend path to extracted data.yaml.');
		} else {
			notify('Dataset path copied into Train');
		}
	}

	function useDatasetPathForTraining() {
		const path = datasetRootPath.trim();
		if (!path) {
			const asset = selectedDatasetAsset();
			if (asset) {
				useDatasetAsset(asset);
				return;
			}
			notify('Enter a backend-visible data.yaml or dataset zip path');
			return;
		}
		datasetPath = path;
		reconcileAssetSelections();
		setTab('Train');
		notify('Dataset path sent to Train');
	}

	function toggleMergeAsset(id) {
		mergeAssetIds = mergeAssetIds.includes(id)
			? mergeAssetIds.filter((assetId) => assetId !== id)
			: [...mergeAssetIds, id];
	}

	function addClassMapping() {
		const source = mappingSourceClass.trim();
		const target = mappingTargetClass.trim();
		if (!source || !target) {
			notify('Enter both old and new class names');
			return;
		}
		classMappings = [
			...classMappings.filter((mapping) => mapping.source !== source),
			{ source, target }
		];
		mappingSourceClass = '';
		mappingTargetClass = '';
		notify('Class mapping added');
	}

	function removeClassMapping(source) {
		classMappings = classMappings.filter((mapping) => mapping.source !== source);
	}

	function datasetManifest() {
		const assets = datasetAssets().filter((file) => mergeAssetIds.includes(file.id));
		const selected = selectedDatasetAsset();
		return {
			name: datasetManifestName.trim() || 'traffic-dataset',
			root_path: datasetRootPath.trim() || selected?.path || '',
			training_path: datasetPath,
			backend_url: backendUrl,
			created_at: new Date().toISOString(),
			assets: assets.length ? assets : selected ? [selected] : [],
			mappings: classMappings
		};
	}

	function saveDatasetManifest() {
		const manifest = { ...datasetManifest(), id: crypto.randomUUID?.() || String(Date.now()) };
		if (!manifest.root_path && !manifest.assets.length) {
			notify('Select assets or enter a backend dataset path first');
			return;
		}
		savedDatasets = [manifest, ...savedDatasets.filter((item) => item.name !== manifest.name)];
		localStorage.setItem('savedDatasets', JSON.stringify(savedDatasets));
		datasetPath = manifest.root_path || datasetPath;
		notify('Dataset manifest saved');
	}

	function useSavedDataset(manifest) {
		datasetManifestName = manifest.name;
		datasetRootPath = manifest.root_path || '';
		datasetPath = manifest.root_path || manifest.training_path || '';
		mergeAssetIds = (manifest.assets || []).map((asset) => asset.id).filter(Boolean);
		classMappings = manifest.mappings || [];
		reconcileAssetSelections();
		setTab('Train');
		notify(`Loaded ${manifest.name} for training`);
	}

	function downloadDatasetManifest(manifest = datasetManifest()) {
		const blob = new Blob([JSON.stringify(manifest, null, 2)], { type: 'application/json' });
		const url = URL.createObjectURL(blob);
		const link = document.createElement('a');
		link.href = url;
		link.download = `${manifest.name || 'traffic-dataset'}-manifest.json`;
		link.click();
		URL.revokeObjectURL(url);
	}

	function addTask() {
		setTab('Inference');
		notify('Configure a traffic inference task');
	}

	function removeTask(index) {
		const task = tasks[index];
		if (task?.status !== 'Completed' && task?.status !== 'Failed')
			send('delete_inference_task', { id: task.id });
		tasks = tasks.filter((_, taskIndex) => taskIndex !== index);
	}

	function completedCount() {
		return tasks.filter((task) => task.status === 'Completed').length;
	}

	function failedCount() {
		return tasks.filter((task) => task.status === 'Failed').length;
	}
</script>

<svelte:head>
	<title>SAM2YOLO Academics</title>
	<meta
		name="description"
		content="A working SAM2YOLO-style academic workflow interface for imports, datasets, inference, training, and exports."
	/>
</svelte:head>

<main class="page-shell">
	<section class="browser-top" aria-label="Prototype frame">
		<button class="menu-button" aria-label="Open menu" onclick={() => (showMenu = !showMenu)}>
			<span></span>
			<span></span>
			<span></span>
		</button>
		<h1>Academics</h1>
		<div class="brand-mark" aria-hidden="true">S2Y</div>
	</section>

	<section class="workspace">
		{#if showMenu}
			<nav class="mobile-menu" aria-label="Mobile navigation">
				{#each tabs as tab (tab)}
					<button class:active={activeTab === tab} onclick={() => setTab(tab)}>{tab}</button>
				{/each}
			</nav>
		{/if}

		<div class="app-window">
			<header class="window-bar">
				<span></span>
				<strong>SAM2YOLO</strong>
				<button aria-label="Close window">x</button>
			</header>

			<div class="app-body">
				<aside class="sidebar" aria-label="Primary navigation">
					{#each tabs as tab (tab)}
						<button class:active={activeTab === tab} onclick={() => setTab(tab)}>{tab}</button>
					{/each}
				</aside>

				<section class="content-panel">
					{#if activeTab === 'Overview'}
						<div class="metrics">
							<div>
								<span>Remotes</span>
								<strong>{remotes.length}</strong>
							</div>
							<div>
								<span>Tasks</span>
								<strong>{tasks.length}</strong>
							</div>
							<div>
								<span>Completed</span>
								<strong>{completedCount()}</strong>
							</div>
							<div>
								<span>Failed</span>
								<strong>{failedCount()}</strong>
							</div>
						</div>

						<div class="section-title-row">
							<h2>Tasks</h2>
							<button class="icon-action" aria-label="Add task" onclick={addTask}>+</button>
						</div>

						<div class="task-list">
							{#each tasks as task, index (task.id)}
								<div class="task-row">
									<strong>{task.name}</strong>
									<div class="progress-track" aria-label={`${task.name} progress`}>
										<span style={`width: ${task.progress}%`}></span>
									</div>
									<small>
										{task.status}
										{#if task.chunks?.length}
											<br />{task.chunks.length} result chunk{task.chunks.length === 1 ? '' : 's'}
										{/if}
									</small>
									<button aria-label={`Delete ${task.name}`} onclick={() => removeTask(index)}
										>trash</button
									>
								</div>
							{/each}
						</div>

						{#if liveResults.length}
							<div class="section-title-row">
								<h2>Live results</h2>
								<span class="status-pill">{liveResults.length} frames · {totalDetections} det</span>
							</div>
							<div class="result-grid">
								{#each liveResults as item (item.key)}
									<figure class="result-tile">
										<canvas use:drawResult={{ result: item.result, imgUrl: item.imgUrl }}></canvas>
										<figcaption>
											<span class="result-class">{item.className}</span>
											<span class="result-badge" class:zero={!item.detections}
												>{item.detections}</span
											>
										</figcaption>
									</figure>
								{/each}
							</div>
						{/if}
					{:else if activeTab === 'Remotes'}
						<div class="section-title-row">
							<h2>Remote Notebook</h2>
							<span class="status-pill">{fleetBusy ? 'Creating room' : 'Single user room'}</span>
						</div>

						<div class="form-group">
							<span class="field-label">Backend URL</span>
							<div class="input-action">
								<input bind:value={backendUrl} placeholder="http://127.0.0.1:8000" />
								<button type="button" onclick={saveBackendUrl}>Save</button>
							</div>
							<small>
								Status: {connectionState} · Model: {modelState}
								{modelReady ? 'ready' : 'not ready'}
							</small>
						</div>

						<div class="remote-card fleet-card">
							<div class="section-title-row">
								<div>
									<h3>Notebook setup</h3>
									<p>
										Copy the cell into your notebook and keep it running. Reloading this page keeps
										the same room.
									</p>
								</div>
								<span class="status-pill">{reachableWorkers.length || 0} connected</span>
							</div>

							{#if fleetBusy && !fleetRoomId}
								<button class="primary-action" type="button" disabled> Creating room… </button>
							{:else if fleetRoomId}
								<label>
									Worker name
									<span class="input-action">
										<input
											bind:value={fleetWorkerName}
											placeholder="nimble-sloth-9341"
											onchange={saveFleetSession}
										/>
										<button type="button" onclick={regenerateWorkerName}>↻</button>
									</span>
								</label>
								<label class="code-label">
									Terminal command
									<textarea readonly value={fleetCommand()}></textarea>
								</label>

								<div class="remote-actions">
									<button class="ghost-button" type="button" onclick={copyFleetNotebookCell}
										>Copy notebook cell</button
									>
									<button class="ghost-button" type="button" onclick={copyFleetCommand}
										>Copy terminal command</button
									>
									<button class="ghost-button" type="button" onclick={regenerateWorkerName}
										>New name</button
									>
									<button class="ghost-button" type="button" onclick={refreshFleetWorkers}
										>Refresh workers</button
									>
								</div>
								<small class="fleet-hint"
									>The notebook-cell button runs this terminal command through a Python
									pseudo-terminal.</small
								>
							{:else}
								<button class="primary-action" type="button" onclick={() => createFleetRoom()}>
									Create room
								</button>
							{/if}

							{#if fleetError}
								<small class="fleet-error">{fleetError}</small>
							{/if}

							{#if fleetWorkers.length}
								<div class="remote-list">
									{#each fleetWorkers as worker (worker.tunnel_id)}
										{@const reachable = isWorkerReachable(worker)}
										<div class="remote-row worker-row">
											<input
												type="checkbox"
												aria-label={`Include ${worker.name}`}
												disabled={!reachable}
												checked={reachable && !excludedWorkerIds.includes(worker.tunnel_id)}
												onchange={() => toggleWorker(worker)}
											/>
											<span class="status-dot" class:online={reachable}></span>
											<strong class="mono">{worker.name}</strong>
											<span>{reachable ? 'Reachable' : worker.status || 'Unreachable'}</span>
											<button type="button" onclick={() => connectToWorker(worker)}>Connect</button>
										</div>
									{/each}
								</div>
							{/if}
						</div>
					{:else if activeTab === 'Import'}
						<div class="import-layout">
							<div class="tab-strip">
								<button
									class:selected={importMode === 'upload'}
									onclick={() => (importMode = 'upload')}>Upload</button
								>
								<button
									class:selected={importMode === 'drive'}
									onclick={() => (importMode = 'drive')}>Google Drive</button
								>
							</div>
							{#if importMode === 'upload'}
								<div class="upload-card">
									<p>Upload images or video<br />from local device</p>
									<span>JPG, PNG · MP4, MOV, WEBM, AVI, MKV, DAV</span>
									<input
										bind:this={localFileInput}
										class="hidden-file-input"
										type="file"
										accept={localMediaAccept}
										multiple
										onchange={handleLocalFileChange}
									/>
									<button disabled={uploadBusy} onclick={chooseLocalFiles}>
										{uploadBusy ? 'Uploading...' : 'Upload'}
									</button>
								</div>
							{:else}
								<div class="upload-card drive-card">
									<p>Upload video files<br />from Google Drive</p>
									<span>Public Drive file link</span>
									<div class="drive-upload">
										<input bind:value={googleDriveUrl} placeholder="Google Drive share link" />
										<button disabled={uploadBusy} onclick={uploadFromGoogleDrive}>
											{uploadBusy ? 'Uploading...' : 'Upload'}
										</button>
									</div>
								</div>
							{/if}
							{#if uploadProgress.visible}
								<div class="upload-progress" aria-live="polite">
									<div class="upload-progress-head">
										<strong>{uploadProgress.label}</strong>
										<span>{uploadProgress.value}%</span>
									</div>
									<div
										class="progress-track upload-progress-track"
										role="progressbar"
										aria-valuemin="0"
										aria-valuemax="100"
										aria-valuenow={uploadProgress.value}
										aria-label={`${uploadProgress.label} progress`}
									>
										<span style={`width: ${uploadProgress.value}%`}></span>
									</div>
									<small>{uploadProgress.detail}</small>
								</div>
							{/if}
						</div>

						<h2>Imports</h2>
						{#if uploadedImports.length}
							<div class="import-list">
								{#each uploadedImports as file (file.id)}
									<div class="single-row">
										<span class="import-name">{file.name}<br /><small>{file.id}</small></span>
										<button aria-label={`Remove ${file.name}`} onclick={() => removeImport(file)}
											>minus</button
										>
									</div>
								{/each}
							</div>
						{:else}
							<p class="empty-imports">No uploaded video files yet.</p>
						{/if}
					{:else if activeTab === 'Inference'}
						<div class="model-switch">
							<span class="field-label">Inference model</span>
							<div class="seg-control">
								<button
									class:active={inferenceModel === 'sam'}
									onclick={() => (inferenceModel = 'sam')}>SAM 3.1</button
								>
								<button
									class:active={inferenceModel === 'yolo'}
									onclick={() => (inferenceModel = 'yolo')}>YOLO</button
								>
							</div>
							<small>
								{inferenceModel === 'sam'
									? 'Text-prompted segmentation. Each prompt below becomes a dataset class.'
									: 'COCO-class detection/segmentation. Prompts map to vehicle classes.'}
							</small>
							{#if inferenceModel === 'sam'}
								<div class="settings-grid compact sam-params">
									<label
										>Video sample FPS
										<input type="number" bind:value={samFps} min="0.5" step="0.5" /></label
									>
									<label
										>Max frames (0 = all)
										<input type="number" bind:value={samMaxFrames} min="0" step="1" /></label
									>
									<label>Batch <input type="number" bind:value={samBatch} min="1" step="1" /></label
									>
								</div>
								<small class="fleet-distribute">
									{#if selectedWorkers.length}
										Will distribute across {selectedWorkers.length} worker{selectedWorkers.length ===
										1
											? ''
											: 's'} (manage in Remotes → Fleet)
									{:else}
										No fleet workers selected — will run on the single Backend URL. Add workers in
										Remotes → Fleet.
									{/if}
								</small>
							{/if}
						</div>

						<div class="split-grid">
							<div>
								<h2>{inferenceModel === 'sam' ? 'Text prompts (classes)' : 'Prompts'}</h2>
								<div class="form-group">
									<span class="field-label">Create new</span>
									{#if inferenceModel === 'sam'}
										<div class="input-action">
											<input
												bind:value={newPromptText}
												placeholder="e.g. person, delivery truck"
												onkeydown={(e) => e.key === 'Enter' && addPrompt()}
											/>
											<button disabled={!newPromptText.trim()} onclick={addPrompt}>Add</button>
										</div>
									{:else}
										<div class="input-action">
											<select bind:value={selectedPromptType}>
												<option value="" disabled>Select type</option>
												{#each vehicleTypes as v (v)}
													<option value={v}>{v}</option>
												{/each}
											</select>
											<button
												disabled={!selectedPromptType}
												onclick={() => {
													if (selectedPromptType && !promptTags.includes(selectedPromptType)) {
														promptTags = [...promptTags, selectedPromptType];
													}
													selectedPromptType = '';
												}}>Add</button
											>
										</div>
									{/if}
									{#if promptTags.length}
										<div class="chips">
											{#each promptTags as tag (tag)}
												<span
													>{tag}
													<button
														aria-label={`Remove ${tag}`}
														onclick={() => (promptTags = promptTags.filter((t) => t !== tag))}
														>x</button
													></span
												>
											{/each}
										</div>
									{/if}
								</div>
								<div class="form-group">
									<span class="field-label">Save prompts (optional)</span>
									<input bind:value={promptSaveName} placeholder="Name" />
									<div style="margin-top:8px;display:flex;gap:8px;align-items:center;">
										<button class="ghost-button" onclick={savePrompt}>Save</button>
										{#if savedPrompts.length}
											<select
												onchange={(e) => {
													const sel = JSON.parse(e.target.value || 'null');
													if (sel) {
														promptTags = sel.tags || [];
														notify(`Loaded prompt ${sel.name}`);
													}
												}}
											>
												<option value="">Load saved</option>
												{#each savedPrompts as p (p.name)}
													<option value={JSON.stringify(p)}>{p.name}</option>
												{/each}
											</select>
										{/if}
									</div>
								</div>
							</div>

							<div>
								<h3>Or use existing</h3>
								<input placeholder="Search" />
								<div class="chips catalog">
									<span>Defaults</span>
									<span>All Vehicle</span>
									<span>Road Elements</span>
								</div>
							</div>
						</div>

						<div class="divider"></div>

						<div class="section-title-row">
							<h2>Select Imports</h2>
							<button class="icon-action" aria-label="Add import" onclick={() => setTab('Import')}
								>+</button
							>
						</div>
						<select bind:value={selectedImportId} disabled={!mediaAssets().length}>
							{#if mediaAssets().length}
								{#each mediaAssets() as file (file.id)}
									<option value={file.id}>{file.name}</option>
								{/each}
							{:else}
								<option value="">No uploaded image/video imports</option>
							{/if}
						</select>
						{#if !mediaAssets().length}
							<p class="empty-note">
								Import a road video first. The Start button sends that backend file ID into YOLO
								inference.
							</p>
						{/if}

						<div class="settings-grid">
							<label
								>Confidence <input
									bind:value={confidence}
									min="0"
									max="1"
									step="0.01"
									type="number"
								/></label
							>
							<div>
								<label
									>IoU <input bind:value={iou} min="0" max="1" step="0.01" type="number" /></label
								>
							</div>
							<label
								>Image Size <input
									bind:value={inferenceImageSize}
									min="320"
									step="32"
									type="number"
								/></label
							>
							<label
								>Frame keep rate <input
									bind:value={frameKeepRate}
									min="0"
									max="1"
									step="0.01"
									type="number"
								/></label
							>
							<label
								>Batch <input bind:value={inferenceBatch} min="1" step="1" type="number" /></label
							>
						</div>

						<div class="section-footer">
							<label class="check-line"
								><input bind:checked={temporalDownsampling} type="checkbox" /> Temporal Downsampling</label
							>
							<button
								class="primary-action"
								disabled={!mediaAssets().length}
								onclick={startInference}>Start</button
							>
						</div>
					{:else if activeTab === 'Dataset'}
						<div class="section-title-row dataset-hero">
							<div>
								<span class="eyebrow">Dataset</span>
								<h2>Training data</h2>
							</div>
							<button
								class="cloud-button"
								aria-label="Refresh backend files"
								onclick={() => send('list_files')}>sync</button
							>
						</div>

						<div class="dataset-summary">
							<div>
								<strong>{datasetAssets().length}</strong>
								<span>backend assets</span>
							</div>
							<div>
								<strong>{mergeAssetIds.length}</strong>
								<span>in manifest</span>
							</div>
							<div>
								<strong>{savedDatasets.length}</strong>
								<span>saved manifests</span>
							</div>
						</div>

						<div class="dataset-flow">
							<section class="dataset-card">
								<h3>Add files</h3>
								<div class="tab-strip">
									{#each datasetImportModes as mode (mode)}
										<button
											class:selected={datasetMode === mode}
											onclick={() => (datasetMode = mode)}>{mode}</button
										>
									{/each}
								</div>

								{#if datasetMode === 'upload'}
									<input
										bind:this={datasetFileInput}
										class="hidden-file-input"
										type="file"
										accept={datasetAccept}
										multiple
										onchange={handleDatasetFileChange}
									/>
									<button
										class="primary-action slim"
										disabled={uploadBusy}
										onclick={chooseDatasetFiles}
									>
										{uploadBusy ? 'Uploading...' : 'Choose files'}
									</button>
								{:else if datasetMode === 'drive'}
									<div class="stacked-actions">
										<input bind:value={googleDriveUrl} placeholder="Google Drive dataset link" />
										<button
											class="ghost-button"
											disabled={uploadBusy}
											onclick={downloadDatasetFromGoogleDrive}
										>
											{uploadBusy ? 'Downloading...' : 'Download from Drive'}
										</button>
									</div>
								{:else}
									<div class="stacked-actions">
										<input
											bind:value={datasetDirectUrl}
											placeholder="https://example.com/dataset.zip"
										/>
										<button
											class="ghost-button"
											disabled={uploadBusy}
											onclick={downloadDatasetDirectUrl}
										>
											{uploadBusy ? 'Downloading...' : 'Download URL'}
										</button>
									</div>
								{/if}
								{#if uploadProgress.visible}
									<div class="upload-progress compact" aria-live="polite">
										<div class="upload-progress-head">
											<strong>{uploadProgress.label}</strong>
											<span>{uploadProgress.value}%</span>
										</div>
										<div
											class="progress-track upload-progress-track"
											role="progressbar"
											aria-valuemin="0"
											aria-valuemax="100"
											aria-valuenow={uploadProgress.value}
											aria-label={`${uploadProgress.label} progress`}
										>
											<span style={`width: ${uploadProgress.value}%`}></span>
										</div>
										<small>{uploadProgress.detail}</small>
									</div>
								{/if}
							</section>

							<section class="dataset-card">
								<h3>Path</h3>
								<label
									>Dataset path
									<input
										bind:value={datasetRootPath}
										placeholder="/kaggle/working/dataset/data.yaml"
									/></label
								>
								<label>Manifest name <input bind:value={datasetManifestName} /></label>
								<div class="remote-actions dataset-actions">
									<button class="ghost-button" type="button" onclick={saveDatasetManifest}
										>Save</button
									>
									<button
										class="primary-action slim"
										type="button"
										onclick={useDatasetPathForTraining}>Use in Train</button
									>
								</div>
							</section>
						</div>

						<section class="dataset-card">
							<div class="section-title-row">
								<h3>Backend files</h3>
								<button
									class="ghost-button"
									type="button"
									onclick={() => (mergeAssetIds = datasetAssets().map((file) => file.id))}
									>Select all</button
								>
							</div>
							{#if datasetAssets().length}
								<div class="dataset-asset-list friendly">
									{#each datasetAssets() as file (file.id)}
										<div class:selected={selectedDatasetId === file.id} class="dataset-asset-row">
											<label class="check-line">
												<input
													checked={mergeAssetIds.includes(file.id)}
													type="checkbox"
													onchange={() => toggleMergeAsset(file.id)}
												/>
												<span>
													<strong>{file.name}</strong>
													<small>{fileKind(file)} · {file.path}</small>
												</span>
											</label>
											<div class="row-actions">
												<button type="button" onclick={() => useDatasetAsset(file)}>use path</button
												>
												<button type="button" onclick={() => removeImport(file)}>delete</button>
											</div>
										</div>
									{/each}
								</div>
							{:else}
								<p class="empty-imports">No dataset files yet.</p>
							{/if}
						</section>

						<details class="advanced-card">
							<summary>Advanced: class mapping and manifest export</summary>
							<div class="split-grid dataset-tools">
								<div>
									<h3>Class mapping notes</h3>
									<div class="settings-grid compact mapping-inputs">
										<label
											>Old class <input
												bind:value={mappingSourceClass}
												placeholder="Red Motor Bike"
											/></label
										>
										<label
											>New class <input
												bind:value={mappingTargetClass}
												placeholder="motorbike_red"
											/></label
										>
									</div>
									<button class="ghost-button wide" type="button" onclick={addClassMapping}
										>Add mapping</button
									>
									{#if classMappings.length}
										<div class="map-table">
											<strong>Old</strong>
											<strong>New</strong>
											{#each classMappings as mapping (mapping.source)}
												<span>{mapping.source}</span>
												<span>
													{mapping.target}
													<button
														aria-label={`Remove mapping ${mapping.source}`}
														type="button"
														onclick={() => removeClassMapping(mapping.source)}>x</button
													>
												</span>
											{/each}
										</div>
									{/if}
								</div>
								<div>
									<h3>Saved manifests</h3>
									<button
										class="ghost-button"
										type="button"
										onclick={() => downloadDatasetManifest()}>Download current manifest</button
									>
									{#if savedDatasets.length}
										<div class="saved-manifest-list">
											{#each savedDatasets as manifest (manifest.id)}
												<div>
													<strong>{manifest.name}</strong>
													<small
														>{manifest.root_path ||
															`${manifest.assets?.length || 0} asset(s)`}</small
													>
													<button type="button" onclick={() => useSavedDataset(manifest)}
														>use</button
													>
												</div>
											{/each}
										</div>
									{:else}
										<p class="empty-imports">No saved manifests yet.</p>
									{/if}
								</div>
							</div>
						</details>
					{:else if activeTab === 'Train'}
						<div class="section-title-row train-hero">
							<div>
								<span class="eyebrow">Train</span>
								<h2>YOLO training</h2>
							</div>
							<button
								class="cloud-button"
								aria-label="Refresh backend files"
								onclick={() => send('list_files')}>sync</button
							>
						</div>

						<div class="train-flow">
							<section class="dataset-card">
								<h3>Dataset</h3>
								<label
									>YOLO data path
									<input
										bind:value={datasetPath}
										placeholder="/kaggle/working/dataset/data.yaml"
									/></label
								>
								<button class="ghost-button" type="button" onclick={() => setTab('Dataset')}
									>Dataset tab</button
								>
							</section>

							<section class="dataset-card">
								<h3>Media</h3>
								<label
									>Anchor media
									<select bind:value={selectedTrainingAnchorId} disabled={!mediaAssets().length}>
										{#if mediaAssets().length}
											{#each mediaAssets() as file (file.id)}
												<option value={file.id}>{file.name}</option>
											{/each}
										{:else}
											<option value="">Upload one image or video first</option>
										{/if}
									</select></label
								>
								<button class="ghost-button" type="button" onclick={() => setTab('Import')}
									>Upload media</button
								>
							</section>

							<section class="dataset-card">
								<h3>Model</h3>
								<label
									>Variant
									<select bind:value={modelVariant}>
										<option value="yolo11n">yolo11n · fastest</option>
										<option value="yolo11s">yolo11s · balanced</option>
										<option value="yolo11m">yolo11m · heavier</option>
									</select></label
								>
								<label>Run name <input bind:value={trainingName} /></label>
							</section>
						</div>

						<div class="train-grid train-params">
							<div>
								<h3>Training parameters</h3>
								<label>Batch Size <input bind:value={trainingBatch} step="1" type="number" /></label
								>
								<label
									>Image Size <input
										bind:value={trainingImageSize}
										min="320"
										step="32"
										type="number"
									/></label
								>
							</div>
							<div>
								<h3>Compute</h3>
								<label
									>Epoch <input bind:value={trainingEpochs} min="1" step="1" type="number" /></label
								>
								<label
									>Workers <input
										bind:value={trainingWorkers}
										min="0"
										step="1"
										type="number"
									/></label
								>
							</div>
						</div>

						<div class="settings-grid compact train-output">
							<div>
								<h3>Output</h3>
								<div class="segmented">
									<button class="selected">Local Drive</button>
									<button onclick={() => (showMegaModal = true)}>Mega</button>
								</div>
							</div>
							<div>
								<button class="ghost-button" onclick={() => (showMegaModal = true)}
									>Select path</button
								>
								<label>Project Folder <input bind:value={trainingProject} /></label>
							</div>
							<div>
								<h3>Machine</h3>
								<div class="segmented">
									<button class="selected">Local Machine</button>
									<button>Remote</button>
								</div>
							</div>
							<label
								>Device <input bind:value={trainingDevice} placeholder="0, cpu, or cuda:0" /></label
							>
						</div>

						<button
							class="primary-action centered"
							disabled={!selectedTrainingAnchorId ||
								!(datasetPath.trim() || datasetRootPath.trim())}
							onclick={startTraining}>Start</button
						>
					{:else if activeTab === 'Export'}
						<div class="remote-card">
							<div class="section-title-row">
								<div>
									<h3>Compile YOLO dataset</h3>
									<p>
										Turn the streamed SAM results into a YOLO-compatible dataset (images + labels +
										data.yaml). Each prompt becomes a class.
									</p>
								</div>
								<span class="status-pill">{liveResults.length} frames</span>
							</div>
							<div class="export-format">
								<span class="field-label">Export format</span>
								<div class="seg-control">
									<button
										class:active={exportFormat === 'detect'}
										type="button"
										onclick={() => (exportFormat = 'detect')}>Detection boxes</button
									>
									<button
										class:active={exportFormat === 'segment'}
										type="button"
										onclick={() => (exportFormat = 'segment')}>Segmentation polygons</button
									>
								</div>
								<small>
									{exportFormat === 'segment'
										? 'YOLO-seg labels: cls x1 y1 x2 y2 … from mask polygons (box fallback if no mask).'
										: 'YOLO detection labels: cls cx cy w h from SAM boxes.'}
								</small>
							</div>
							<div class="settings-grid compact">
								<label
									>Dataset name <input
										bind:value={datasetManifestName}
										placeholder="sam-dataset"
									/></label
								>
								<div class="dataset-classes">
									<span class="field-label">Classes</span>
									<div class="chips">
										{#each promptTags as tag, i (tag)}
											<span>{i}: {tag}</span>
										{/each}
										{#if !promptTags.length}<small>Add prompts in the Inference tab</small>{/if}
									</div>
								</div>
							</div>
							<div class="remote-actions">
								<button
									class="primary-action"
									type="button"
									disabled={exportingDataset || !liveResults.length || !promptTags.length}
									onclick={exportYoloDataset}
								>
									{exportingDataset ? 'Compiling…' : 'Export dataset (.zip)'}
								</button>
							</div>
							{#if datasetStats}
								<small class="dataset-stat">
									Last export ({datasetStats.format === 'segment' ? 'segment' : 'detect'}): {datasetStats.frameCount}
									frames · {datasetStats.labeledFrames} labeled · {datasetStats.boxesTotal}
									{datasetStats.format === 'segment' ? 'polygons' : 'boxes'} · {datasetStats.classes}
									classes
								</small>
							{/if}
						</div>

						<div class="export-toolbar">
							<input placeholder="Search" />
							<select><option>All</option><option>Models</option><option>Datasets</option></select>
						</div>
						<div class="export-list">
							{#each tasks.filter((task) => task.status === 'Completed') as task (task.id)}
								<div>
									<strong>{task.name}</strong>
									<span>{task.resultPath || `${task.chunks?.length || 0} chunk(s)`}</span>
									{#if task.chunks?.length}
										<button onclick={() => downloadResult(task.chunks[0])}>download</button>
									{:else}
										<button aria-label={`Download ${task.name}`} disabled>download</button>
									{/if}
								</div>
							{/each}
							{#if !tasks.some((task) => task.status === 'Completed')}
								<div>
									<strong>No completed backend artifacts yet</strong>
									<span>Run inference or training first</span>
									<button disabled>download</button>
								</div>
							{/if}
						</div>
					{:else}
						<div class="about-panel">
							<h2>SAM2YOLO</h2>
							<p>
								A compact academic workflow for importing road videos, preparing datasets, running
								prompts, training models, and exporting artifacts.
							</p>
							<button class="primary-action" onclick={() => setTab('Overview')}
								>Open overview</button
							>
						</div>
					{/if}
				</section>
			</div>
		</div>
	</section>

	{#if showMegaModal}
		<div class="modal-backdrop" role="presentation">
			<div
				class="modal"
				role="dialog"
				aria-modal="true"
				aria-label="Mega credentials"
				tabindex="-1"
			>
				<button class="modal-close" aria-label="Close modal" onclick={() => (showMegaModal = false)}
					>x</button
				>
				<h2>Enter mega credentials</h2>
				<label>Email <input value={'dataset_from_{imports}'} /></label>
				<label>Password <input type="password" value={'dataset_from_{imports}'} /></label>
				<button
					class="primary-action"
					onclick={() => {
						showMegaModal = false;
						notify('Mega credentials saved');
					}}
				>
					Save
				</button>
			</div>
		</div>
	{/if}

	{#if toast}
		<div class="toast" role="status">
			<button aria-label="Dismiss notification" onclick={() => (toast = '')}>x</button>
			<strong>{toast.split('!')[0]}{toast.includes('!') ? '!' : ''}</strong>
			<span>{toast.includes('!') ? toast.split('!').slice(1).join('!').trim() : ''}</span>
		</div>
	{/if}
</main>

<style>
	:global(*) {
		box-sizing: border-box;
	}

	:global(body) {
		margin: 0;
		min-width: 320px;
		background: #1f1f1f;
		font-family:
			Inter,
			ui-sans-serif,
			system-ui,
			-apple-system,
			BlinkMacSystemFont,
			'Segoe UI',
			sans-serif;
		color: #181818;
	}

	button,
	input,
	select,
	textarea {
		font: inherit;
	}

	button {
		cursor: pointer;
	}

	.page-shell {
		min-height: 100vh;
		background:
			radial-gradient(circle at 20% 12%, rgba(49, 91, 151, 0.3), transparent 24rem),
			linear-gradient(180deg, #111 0, #212121 38%, #191919 100%);
		padding-bottom: 32px;
	}

	.browser-top {
		position: sticky;
		top: 0;
		z-index: 10;
		display: grid;
		grid-template-columns: 56px 1fr 56px;
		align-items: center;
		height: 78px;
		background: #2b2b2b;
		color: #f4f4f4;
		box-shadow: 0 1px 0 rgba(255, 255, 255, 0.05);
	}

	.browser-top h1 {
		margin: 0;
		text-align: center;
		font-size: clamp(1.05rem, 2vw, 1.5rem);
		font-weight: 500;
		letter-spacing: 0;
	}

	.menu-button {
		display: grid;
		gap: 5px;
		width: 42px;
		height: 42px;
		margin-left: 14px;
		padding: 10px;
		border: 0;
		background: transparent;
	}

	.menu-button span {
		display: block;
		height: 2px;
		background: #d9d9d9;
	}

	.brand-mark {
		justify-self: center;
		width: 34px;
		height: 34px;
		display: grid;
		place-items: center;
		border-radius: 50%;
		background: #111;
		color: #d86f32;
		font-size: 0.63rem;
		font-weight: 800;
	}

	.workspace {
		position: relative;
		display: grid;
		place-items: center;
		min-height: calc(100vh - 78px);
		padding: clamp(24px, 6vw, 74px) 16px;
		overflow: hidden;
	}

	.mobile-menu {
		position: fixed;
		top: 78px;
		left: 14px;
		z-index: 12;
		display: grid;
		gap: 3px;
		width: min(220px, calc(100vw - 28px));
		padding: 10px;
		border: 1px solid rgba(255, 255, 255, 0.1);
		border-radius: 8px;
		background: #f5f5f5;
		box-shadow: 0 18px 60px rgba(0, 0, 0, 0.28);
	}

	.mobile-menu button {
		border: 0;
		border-left: 4px solid transparent;
		background: transparent;
		padding: 9px 10px;
		text-align: left;
	}

	.mobile-menu .active {
		border-left-color: #d86f32;
		background: #fff;
		font-weight: 700;
	}

	.app-window {
		width: min(1080px, 94vw);
		min-height: min(700px, calc(100vh - 150px));
		background: #f4f4f4;
		border: 1px solid #d8d8d8;
		border-radius: 8px;
		overflow: hidden;
		box-shadow: 0 34px 90px rgba(0, 0, 0, 0.38);
	}

	.window-bar {
		display: grid;
		grid-template-columns: 42px 1fr 42px;
		align-items: center;
		height: 26px;
		border-bottom: 1px solid #dbdbdb;
		background: #fff;
	}

	.window-bar strong {
		text-align: center;
		font-size: 0.75rem;
	}

	.window-bar button {
		border: 0;
		background: transparent;
		color: #282828;
	}

	.app-body {
		display: grid;
		grid-template-columns: 132px 1fr;
		min-height: calc(min(700px, calc(100vh - 150px)) - 26px);
	}

	.sidebar {
		display: grid;
		align-content: start;
		gap: 2px;
		padding: 12px 0;
		border-right: 1px solid #dedede;
		background: #f7f7f7;
	}

	.sidebar button {
		height: 28px;
		border: 0;
		border-left: 4px solid transparent;
		background: transparent;
		padding: 0 14px;
		text-align: left;
		font-size: 0.85rem;
		color: #161616;
	}

	.sidebar button:hover,
	.sidebar .active {
		border-left-color: #d86f32;
		background: #fff;
		font-weight: 700;
	}

	.content-panel {
		padding: clamp(18px, 3vw, 32px);
		background: #f3f3f3;
		overflow: auto;
	}

	h2,
	h3,
	p {
		margin-top: 0;
	}

	h2 {
		margin-bottom: 14px;
		font-size: 1.15rem;
	}

	h3 {
		margin-bottom: 8px;
		font-size: 0.9rem;
	}

	p,
	label,
	small {
		font-size: 0.82rem;
	}

	label,
	.field-label {
		display: grid;
		gap: 6px;
		font-weight: 700;
	}

	input,
	select,
	textarea {
		width: 100%;
		border: 0;
		border-radius: 3px;
		background: #fff;
		padding: 0 12px;
		color: #303030;
		outline: 1px solid transparent;
	}

	input,
	select {
		height: 36px;
	}

	textarea {
		min-height: 128px;
		padding: 12px;
		resize: vertical;
		font-family: 'JetBrains Mono', 'SFMono-Regular', Consolas, 'Liberation Mono', monospace;
		font-size: 0.75rem;
		line-height: 1.5;
	}

	input:focus,
	select:focus,
	textarea:focus {
		outline-color: #d86f32;
	}

	.metrics {
		display: grid;
		grid-template-columns: repeat(4, minmax(110px, 1fr));
		margin-bottom: 38px;
	}

	.metrics div {
		display: grid;
		place-items: center;
		min-height: 102px;
		border-right: 1px solid #d5d5d5;
	}

	.metrics div:last-child {
		border-right: 0;
	}

	.metrics span {
		font-size: 0.95rem;
	}

	.metrics strong {
		color: #d86f32;
		font-size: clamp(2.5rem, 7vw, 4.25rem);
		font-weight: 400;
		line-height: 1;
	}

	.section-title-row {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 16px;
	}

	.icon-action,
	.cloud-button {
		width: 34px;
		height: 34px;
		border: 0;
		border-radius: 4px;
		background: #fff;
		box-shadow: 0 2px 0 #d86f32;
		color: #282828;
	}

	.task-list,
	.remote-list,
	.export-list {
		display: grid;
		gap: 0;
	}

	.remote-card {
		display: grid;
		gap: 16px;
		margin: 18px 0 26px;
		border: 1px solid #ddd;
		border-radius: 10px;
		background: #fff;
		padding: 18px;
		box-shadow: 0 8px 24px rgba(0, 0, 0, 0.04);
	}

	.eyebrow {
		display: inline-block;
		margin-bottom: 4px;
		color: #d86f32;
		font-size: 0.72rem;
		font-weight: 900;
		letter-spacing: 0.08em;
		text-transform: uppercase;
	}

	.status-pill {
		display: inline-grid;
		place-items: center;
		min-height: 28px;
		border-radius: 999px;
		background: #181818;
		padding: 0 12px;
		color: #fff;
		font-size: 0.72rem;
		font-weight: 800;
	}

	.status-dot {
		width: 10px;
		height: 10px;
		border-radius: 50%;
		background: #c9542f;
		box-shadow: 0 0 0 3px rgba(201, 84, 47, 0.18);
	}

	.status-dot.online {
		background: #3fb950;
		box-shadow: 0 0 0 3px rgba(63, 185, 80, 0.2);
	}

	.fleet-error {
		display: block;
		margin-top: 8px;
		color: #f85149;
	}

	.fleet-distribute {
		display: block;
		margin: 8px 0;
		color: #9fb6ff;
	}

	.fleet-hint {
		display: block;
		margin-top: 6px;
		color: #8a8a8a;
	}

	.mono {
		font-family: ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
	}

	.worker-row input[type='checkbox'] {
		accent-color: #3fb950;
		width: auto;
		margin: 0;
	}

	.export-format {
		margin-bottom: 14px;
	}

	.model-switch {
		margin-bottom: 18px;
	}

	.sam-params {
		margin-top: 12px;
	}

	.seg-control {
		display: inline-flex;
		border: 1px solid #2a2a2a;
		border-radius: 10px;
		overflow: hidden;
		margin: 6px 0;
	}

	.seg-control button {
		border: 0;
		background: #141414;
		color: #cfcfcf;
		padding: 8px 18px;
		font-weight: 700;
		cursor: pointer;
	}

	.seg-control button.active {
		background: #2f81f7;
		color: #fff;
	}

	.result-grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
		gap: 12px;
		margin-top: 12px;
	}

	.result-tile {
		margin: 0;
		background: #0f0f0f;
		border: 1px solid #242424;
		border-radius: 12px;
		overflow: hidden;
	}

	.result-tile canvas {
		display: block;
		width: 100%;
		height: auto;
		background: #000;
	}

	.result-tile figcaption {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 6px 10px;
		font-size: 0.78rem;
	}

	.result-class {
		color: #cfcfcf;
		text-transform: capitalize;
	}

	.result-badge {
		background: #2f81f7;
		color: #fff;
		border-radius: 10px;
		padding: 1px 9px;
		font-weight: 800;
	}

	.result-badge.zero {
		background: #3a3a3a;
	}

	.dataset-classes .chips {
		margin-top: 6px;
	}

	.dataset-stat {
		display: block;
		margin-top: 10px;
		color: #8b949e;
	}

	.code-label {
		gap: 8px;
	}

	.remote-actions {
		display: flex;
		flex-wrap: wrap;
		justify-content: flex-end;
		gap: 12px;
	}

	.task-row,
	.remote-row,
	.single-row,
	.export-list div {
		display: grid;
		grid-template-columns: minmax(120px, 1fr) minmax(140px, 1.1fr) minmax(110px, 0.7fr) 52px;
		align-items: center;
		gap: 18px;
		min-height: 45px;
		border-bottom: 2px solid #d86f32;
		font-size: 0.86rem;
	}

	.task-row button,
	.remote-row button,
	.single-row button,
	.export-list button {
		border: 0;
		background: transparent;
		color: #2f2f2f;
		font-size: 0.78rem;
	}

	.progress-track {
		height: 14px;
		background: #dedede;
		overflow: hidden;
	}

	.progress-track span {
		display: block;
		height: 100%;
		background: #d86f32;
	}

	.upload-progress {
		width: min(520px, 100%);
		display: grid;
		gap: 8px;
		color: #555;
	}

	.upload-progress.compact {
		margin-top: 16px;
		width: 100%;
	}

	.upload-progress-head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 16px;
		font-size: 0.82rem;
	}

	.upload-progress-head strong,
	.upload-progress-head span {
		color: #2f2f2f;
	}

	.upload-progress-track {
		width: 100%;
		height: 12px;
	}

	.upload-progress small {
		color: #7a7a7a;
		font-size: 0.75rem;
		overflow-wrap: anywhere;
	}

	.remote-row {
		grid-template-columns: 1px minmax(120px, 1fr) minmax(140px, 1fr) minmax(110px, 1fr) 48px;
	}

	.import-layout {
		display: grid;
		place-items: center;
		gap: 26px;
		margin-bottom: 26px;
	}

	.dataset-hero,
	.train-hero {
		align-items: start;
		margin-bottom: 14px;
	}

	.dataset-summary {
		display: grid;
		grid-template-columns: repeat(3, minmax(0, 1fr));
		gap: 12px;
		margin-bottom: 18px;
	}

	.dataset-summary div {
		border-radius: 10px;
		background: #fff;
		padding: 16px;
		box-shadow: inset 0 -2px 0 rgba(216, 111, 50, 0.35);
	}

	.dataset-summary strong {
		display: block;
		color: #d86f32;
		font-size: 1.8rem;
		line-height: 1;
	}

	.dataset-summary span {
		font-size: 0.78rem;
		color: #555;
	}

	.dataset-flow,
	.train-flow {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 14px;
		margin-bottom: 18px;
	}

	.train-flow {
		grid-template-columns: repeat(3, minmax(0, 1fr));
	}

	.dataset-card {
		position: relative;
		display: grid;
		align-content: start;
		gap: 10px;
		border: 1px solid #ddd;
		border-radius: 12px;
		background: #fff;
		padding: 16px;
		box-shadow: 0 10px 28px rgba(0, 0, 0, 0.045);
	}

	.dataset-card h3 {
		margin: 0;
		font-size: 1rem;
	}

	.dataset-card p {
		margin: 0;
		color: #555;
		line-height: 1.5;
	}

	.slim {
		min-width: 132px;
	}

	.stacked-actions {
		display: grid;
		gap: 10px;
	}

	.advanced-card {
		margin-top: 18px;
		border: 1px solid #ddd;
		border-radius: 12px;
		background: #fff;
		padding: 0;
	}

	.advanced-card summary {
		cursor: pointer;
		padding: 16px 18px;
		font-weight: 800;
	}

	.advanced-card[open] summary {
		border-bottom: 1px solid #eee;
	}

	.advanced-card > .split-grid {
		padding: 18px;
	}

	.tab-strip,
	.segmented {
		display: inline-flex;
		flex-wrap: wrap;
		gap: 1px;
		border-radius: 4px;
		overflow: hidden;
		background: #ebebeb;
	}

	.tab-strip button,
	.segmented button {
		min-height: 34px;
		border: 0;
		background: #fff;
		padding: 0 16px;
		font-size: 0.8rem;
	}

	.tab-strip .selected,
	.segmented .selected {
		background: #d86f32;
		color: #fff;
	}

	.upload-card {
		display: grid;
		place-items: center;
		text-align: center;
		color: #a8a8a8;
	}

	.upload-card p {
		margin: 0;
		font-size: clamp(1.8rem, 5vw, 3rem);
		line-height: 1.08;
	}

	.upload-card span {
		margin: 10px 0 22px;
		font-size: 1.1rem;
	}

	.upload-card button,
	.ghost-button {
		min-width: 136px;
		height: 36px;
		border: 0;
		border-radius: 3px;
		background: #fff;
		box-shadow: 0 2px 0 #d86f32;
		color: #555;
	}

	.upload-card button:disabled {
		cursor: wait;
		opacity: 0.68;
	}

	.hidden-file-input {
		position: absolute;
		width: 1px;
		height: 1px;
		overflow: hidden;
		clip: rect(0 0 0 0);
		white-space: nowrap;
		clip-path: inset(50%);
	}

	.drive-card {
		width: min(560px, 100%);
	}

	.drive-upload {
		display: grid;
		grid-template-columns: 1fr auto;
		gap: 12px;
		width: min(520px, 100%);
	}

	.single-row {
		grid-template-columns: 1fr auto;
	}

	.import-list {
		display: grid;
		gap: 0;
	}

	.import-name {
		color: #181818;
		overflow-wrap: anywhere;
	}

	.empty-imports {
		margin: 0;
		border-bottom: 2px solid #d86f32;
		padding: 14px 0;
		color: #666;
	}

	.split-grid,
	.train-grid,
	.settings-grid {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: clamp(18px, 4vw, 54px);
	}

	.split-grid > div + div {
		border-left: 1px solid #cfcfcf;
		padding-left: clamp(18px, 4vw, 42px);
	}

	.form-group {
		display: grid;
		gap: 10px;
		margin-bottom: 18px;
	}

	.input-action {
		display: grid;
		grid-template-columns: 1fr 36px;
		gap: 6px;
	}

	.input-action button {
		border: 0;
		background: #fff;
		box-shadow: 0 2px 0 #d86f32;
	}

	.chips {
		display: flex;
		flex-wrap: wrap;
		gap: 10px;
	}

	.chips span {
		display: inline-flex;
		align-items: center;
		gap: 8px;
		min-height: 32px;
		border-radius: 3px;
		background: #fff;
		padding: 0 14px;
		font-size: 0.8rem;
	}

	.chips button {
		border: 0;
		background: transparent;
	}

	.divider {
		height: 1px;
		margin: 22px 0;
		background: #d4d4d4;
	}

	.empty-note {
		margin: 48px 0 54px;
		text-align: center;
		font-size: 0.75rem;
		color: #333;
	}

	.settings-grid {
		align-items: end;
		margin-top: 22px;
	}

	.section-footer {
		display: grid;
		grid-template-columns: 1fr auto 1fr;
		align-items: center;
		margin-top: 18px;
	}

	.check-line {
		display: flex;
		align-items: start;
		gap: 8px;
		font-size: 1.02rem;
	}

	.check-line input {
		width: 18px;
		height: 18px;
	}

	.primary-action {
		min-width: 220px;
		height: 44px;
		border: 0;
		border-radius: 3px;
		background: #df7433;
		color: #fff;
		font-size: 1rem;
		font-weight: 800;
	}

	.primary-action:disabled,
	.ghost-button:disabled,
	.upload-card button:disabled {
		cursor: not-allowed;
		opacity: 0.58;
	}

	.dataset-tools {
		margin-bottom: 26px;
	}

	.dataset-asset-list {
		display: grid;
		gap: 10px;
	}

	.dataset-asset-list.friendly {
		max-height: 310px;
		overflow: auto;
		padding-right: 4px;
	}

	.dataset-asset-row {
		display: grid;
		grid-template-columns: minmax(0, 1fr) auto;
		align-items: center;
		gap: 12px;
		border: 1px solid #ddd;
		border-left: 4px solid transparent;
		border-radius: 6px;
		background: #fff;
		padding: 10px;
	}

	.dataset-asset-row.selected {
		border-left-color: #d86f32;
	}

	.dataset-asset-row .check-line {
		align-items: center;
		margin: 0;
	}

	.dataset-asset-row small {
		display: block;
		margin-top: 3px;
		color: #666;
		overflow-wrap: anywhere;
	}

	.row-actions {
		display: flex;
		flex-wrap: wrap;
		justify-content: flex-end;
		gap: 8px;
	}

	.row-actions button {
		border: 0;
		background: transparent;
		color: #2f2f2f;
		font-size: 0.78rem;
	}

	.dataset-actions {
		margin-top: 12px;
	}

	.mapping-inputs {
		gap: 12px;
	}

	.saved-manifest-list {
		display: grid;
		gap: 10px;
		margin-top: 12px;
	}

	.saved-manifest-list div {
		display: grid;
		grid-template-columns: 1fr auto;
		gap: 4px 12px;
		align-items: center;
		border-radius: 6px;
		background: #f6f6f6;
		padding: 10px;
	}

	.saved-manifest-list small {
		color: #666;
		overflow-wrap: anywhere;
	}

	.saved-manifest-list button {
		grid-row: 1 / span 2;
		grid-column: 2;
		border: 0;
		background: transparent;
		color: #2f2f2f;
	}

	.train-params,
	.train-output {
		margin-top: 18px;
	}

	.dataset-tools input {
		margin-bottom: 10px;
	}

	.wide {
		display: block;
		margin: 18px auto;
	}

	.map-table {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 8px 18px;
		margin: 16px 0;
		text-align: center;
		font-size: 0.8rem;
	}

	.map-table span {
		background: #fff;
		padding: 8px;
	}

	.train-grid {
		gap: 20px 54px;
	}

	.train-grid label,
	.settings-grid label {
		margin-bottom: 14px;
	}

	.compact {
		margin-top: 22px;
	}

	.centered {
		display: block;
		margin: 36px auto 0;
	}

	.export-toolbar {
		display: grid;
		grid-template-columns: 1fr 110px;
		gap: 22px;
		margin-bottom: 20px;
	}

	.export-list div {
		grid-template-columns: 1fr minmax(110px, 0.3fr) 48px;
	}

	.about-panel {
		display: grid;
		place-items: center;
		min-height: 430px;
		text-align: center;
	}

	.about-panel p {
		max-width: 520px;
		font-size: 1rem;
		line-height: 1.7;
	}

	.modal-backdrop {
		position: fixed;
		inset: 0;
		z-index: 20;
		display: grid;
		place-items: center;
		background: rgba(0, 0, 0, 0.42);
	}

	.modal,
	.toast {
		position: relative;
		width: min(480px, calc(100vw - 32px));
		border-radius: 12px;
		background: #f4f4f4;
		padding: 26px;
		box-shadow: 0 24px 70px rgba(0, 0, 0, 0.32);
	}

	.modal-close,
	.toast button {
		position: absolute;
		top: 12px;
		right: 14px;
		border: 0;
		background: transparent;
		color: #444;
	}

	.modal label {
		margin-bottom: 14px;
	}

	.toast {
		position: fixed;
		left: 50%;
		bottom: clamp(28px, 8vw, 90px);
		z-index: 30;
		display: grid;
		place-items: center;
		transform: translateX(-50%);
		text-align: center;
	}

	.toast strong {
		margin-bottom: 5px;
		font-size: 1.05rem;
	}

	.toast span {
		max-width: 330px;
		font-size: 0.95rem;
		line-height: 1.35;
	}

	@media (max-width: 760px) {
		.workspace {
			align-items: start;
			padding: 28px 10px 130px;
		}

		.app-window {
			width: 980px;
			min-height: 590px;
			transform: scale(0.58);
			transform-origin: top center;
		}

		.page-shell {
			min-height: 100vh;
		}
	}

	@media (max-width: 560px) {
		.browser-top {
			grid-template-columns: 54px 1fr 54px;
		}

		.app-window {
			transform: scale(0.5);
		}

		.toast {
			bottom: 28px;
		}
	}
</style>
