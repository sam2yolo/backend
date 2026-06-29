<script>
	import { resolve } from '$app/paths';
	import { onDestroy, onMount } from 'svelte';

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
	const defaultKaggleNotebookRef = 'mursalinnasib/notebooka28c21424b';
	const kaggleSetupSteps = [
		'Open your existing Kaggle notebook and make sure GPU acceleration plus internet are enabled.',
		'Run ./create-room-tui locally from this backend folder to create a tunnel room.',
		'Paste the room ID and secret here, then copy the generated cell into that notebook.',
		'When the local room tool prints a public URL, paste that URL below and connect.'
	];

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
	let pendingTask = null;
	let kaggleNotebookRef = $state(defaultKaggleNotebookRef);
	let kaggleRoomId = $state('');
	let kaggleRoomSecret = $state('');
	let kaggleNotebookName = $state('notebooka28c21424b');
	let kaggleTunnelUrl = $state('');

	let uploadedImports = $state([]);
	let selectedImportId = $state('');
	let selectedTrainingAnchorId = $state('');
	let importMode = $state('upload');
	let localFileInput = $state();
	let uploadBusy = $state(false);
	let googleDriveUrl = $state('');

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
					uploadedImports = (payload.files || []).map((file) => ({ ...file, id: file.id }));
					if (!selectedImportId && mediaAssets().length) selectedImportId = mediaAssets()[0].id;
					if (!selectedDatasetId && datasetAssets().length)
						selectedDatasetId = datasetAssets()[0].id;
					if (!selectedTrainingAnchorId && mediaAssets().length)
						selectedTrainingAnchorId = mediaAssets()[0].id;
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
					modelState = payload.model_name || modelVariant;
					notify(`${modelState} is ready`);
					queuePendingTask();
					break;
				case 'model_setup_error':
				case 'model_init_error':
					modelReady = false;
					modelState = 'Failed';
					pendingTask = null;
					notify(errorMessage(payload));
					break;
				case 'file_download_initiated':
					upsertTask(payload.file_id, {
						name: payload.file_name || 'Google Drive import',
						status: 'Downloading',
						progress: 0
					});
					break;
				case 'download_progress':
					upsertTask(payload.file_id, { status: 'Downloading', progress: payload.progress ?? 0 });
					break;
				case 'file_download_completed':
					upsertTask(payload.file_id, { status: 'Completed', progress: 100 });
					googleDriveUrl = '';
					datasetDirectUrl = '';
					uploadBusy = false;
					send('list_files');
					notify('Backend download completed');
					break;
				case 'download_failed':
					upsertTask(payload.file_id, { status: 'Failed' });
					uploadBusy = false;
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
						type: payload.type || 'inference'
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
					break;
				}
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

	function isMediaFile(file) {
		return /\.(jpe?g|png|webp|bmp|mp4|mov|m4v|webm|avi|mkv)$/i.test(file?.name || '');
	}

	function mediaAssets() {
		return uploadedImports.filter(isMediaFile);
	}

	function selectedTrainingAnchor() {
		return mediaAssets().find((file) => file.id === selectedTrainingAnchorId);
	}

	function detectFileType(name = '') {
		return /\.(jpe?g|png|webp|bmp)$/i.test(name) ? 'image' : 'video';
	}

	function fileKind(file) {
		const name = file?.name || '';
		if (/\.(zip)$/i.test(name)) return 'dataset zip';
		if (/\.(ya?ml)$/i.test(name)) return 'data config';
		if (/\.(txt|json|csv)$/i.test(name)) return 'labels';
		if (/\.(jpe?g|png|webp|bmp)$/i.test(name)) return 'image';
		if (/\.(mp4|mov|m4v|webm|avi|mkv)$/i.test(name)) return 'video';
		return 'asset';
	}

	function datasetAssets() {
		return uploadedImports.filter((file) =>
			/\.(zip|ya?ml|txt|json|csv|jpe?g|png|webp|bmp|mp4|mov|m4v|webm|avi|mkv)$/i.test(
				file.name || ''
			)
		);
	}

	function selectedDatasetAsset() {
		return datasetAssets().find((file) => file.id === selectedDatasetId);
	}

	function resultDownloadUrl(chunkId) {
		const params = new URLSearchParams({ id: chunkId, backend: backendUrl });
		return `${resolve('/api/results')}?${params.toString()}`;
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

	function ensureModel(task) {
		if (connectionState !== 'Connected') {
			notify('Connect to the backend first');
			return;
		}
		pendingTask = task;
		if (modelReady) {
			queuePendingTask();
			return;
		}
		modelState = 'Starting';
		send('init_model', { model_name: 'yolo', variant_name: modelVariant });
		notify('Preparing the YOLO environment; the first run can take longer');
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
		send('init_model', { model_name: 'yolo', variant_name: modelVariant });
	}

	function queuePendingTask() {
		if (!pendingTask) return;
		const task = pendingTask;
		pendingTask = null;
		send('create_inference_task', task);
	}

	function startInference() {
		const file = selectedImport();
		if (!file) {
			notify('Select a backend import first');
			return;
		}

		const classes = [
			...new Set(promptTags.map((tag) => cocoVehicleClasses[tag]).filter(Number.isInteger))
		];
		ensureModel({
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
		ensureModel({
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
		connectBackend();
	});

	onDestroy(() => {
		socket?.close?.();
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

	function shellQuote(value) {
		return `'${String(value).replaceAll("'", "'\"'\"'")}'`;
	}

	function kaggleNotebookUrl() {
		const ref = kaggleNotebookRef.trim() || defaultKaggleNotebookRef;
		return `https://www.kaggle.com/code/${ref}/edit`;
	}

	function kaggleBootstrapCode() {
		const notebookName = kaggleNotebookName.trim() || 'kaggle-traffic-gpu';
		if (!kaggleRoomId.trim() || !kaggleRoomSecret.trim()) {
			return [
				'# Local terminal, from the backend project:',
				'./create-room-tui',
				'',
				'# Then paste the generated room ID and secret into this Remotes tab.'
			].join('\n');
		}

		return [
			'# Kaggle notebook cell. Enable GPU + Internet before running.',
			'!wget -q -O run.sh "https://raw.githubusercontent.com/sam2yolo/backend/refs/heads/main/run.sh"',
			`!bash run.sh ${shellQuote(kaggleRoomId.trim())} ${shellQuote(
				kaggleRoomSecret.trim()
			)} ${shellQuote(notebookName)}`
		].join('\n');
	}

	function copyKaggleBootstrap() {
		navigator.clipboard?.writeText(kaggleBootstrapCode());
		notify('Kaggle setup cell copied');
	}

	function openKaggleNotebook() {
		window.open(kaggleNotebookUrl(), '_blank', 'noopener,noreferrer');
	}

	function connectKaggleTunnel() {
		const url = kaggleTunnelUrl.trim().replace(/\/$/, '');
		if (!/^https?:\/\//i.test(url)) {
			notify('Paste the Kaggle tunnel URL, including http:// or https://');
			return;
		}
		backendUrl = url;
		saveBackendUrl();
		notify('Connecting to Kaggle GPU backend');
	}

	function chooseLocalFiles() {
		localFileInput?.click();
	}

	function chooseDatasetFiles() {
		datasetFileInput?.click();
	}

	async function handleLocalFileChange(event) {
		const files = Array.from(event.currentTarget.files ?? []).filter(
			(file) => file.type.startsWith('video/') || /\.(mp4|mov|m4v|webm|avi|mkv)$/i.test(file.name)
		);
		event.currentTarget.value = '';
		if (!files.length) {
			notify('Choose a video file');
			return;
		}

		uploadBusy = true;
		try {
			const body = new FormData();
			for (const file of files) body.append('files', file);
			body.append('backendUrl', backendUrl);
			const response = await fetch('/api/uploads', { method: 'POST', body });
			const result = await response.json();
			if (!response.ok) throw new Error(result.message || 'Upload failed');
			uploadedImports = [...result.files, ...uploadedImports];
			selectedImportId ||= result.files[0]?.id || '';
			selectedTrainingAnchorId ||= result.files.find(isMediaFile)?.id || '';
			notify(
				`${result.files.length} video file${result.files.length === 1 ? '' : 's'} uploaded to backend`
			);
		} catch (error) {
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
			const body = new FormData();
			for (const file of files) body.append('files', file);
			body.append('backendUrl', backendUrl);
			const response = await fetch('/api/uploads', { method: 'POST', body });
			const result = await response.json();
			if (!response.ok) throw new Error(result.message || 'Upload failed');
			uploadedImports = [...result.files, ...uploadedImports];
			selectedImportId ||= result.files.find(isMediaFile)?.id || '';
			selectedDatasetId ||= result.files[0]?.id || '';
			selectedTrainingAnchorId ||= result.files.find(isMediaFile)?.id || '';
			notify(
				`${result.files.length} ${successLabel}${result.files.length === 1 ? '' : 's'} uploaded`
			);
			return result.files;
		} catch (error) {
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
		selectedTrainingAnchorId ||= mediaAssets()[0]?.id || '';
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
		selectedTrainingAnchorId ||= mediaAssets()[0]?.id || '';
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
					{:else if activeTab === 'Remotes'}
						<div class="section-title-row">
							<h2>Remotes</h2>
							<button class="cloud-button" aria-label="Copy remote setup code" onclick={copyCode}
								>cloud</button
							>
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

						<div class="remote-card kaggle-card">
							<div class="section-title-row">
								<div>
									<h3>Kaggle online GPU</h3>
									<p>
										Run this backend in your existing Kaggle notebook, expose port 8000 through the
										repo tunnel, then connect the website to that public URL.
									</p>
								</div>
								<span class="status-pill">GPU remote</span>
							</div>

							<div class="kaggle-steps">
								{#each kaggleSetupSteps as step, index (`kaggle-step-${index}`)}
									<div>
										<strong>{index + 1}</strong>
										<span>{step}</span>
									</div>
								{/each}
							</div>

							<div class="settings-grid compact">
								<label
									>Existing Notebook
									<input
										bind:value={kaggleNotebookRef}
										placeholder="username/notebook-slug"
									/></label
								>
								<label
									>Room ID <input
										bind:value={kaggleRoomId}
										placeholder="from create-room-tui"
									/></label
								>
								<label
									>Room Secret
									<input bind:value={kaggleRoomSecret} placeholder="from create-room-tui" /></label
								>
								<label
									>Notebook Name <input
										bind:value={kaggleNotebookName}
										placeholder="traffic-gpu"
									/></label
								>
								<label
									>Kaggle Tunnel URL
									<input
										bind:value={kaggleTunnelUrl}
										placeholder="http://163.61.236.112:20000"
									/></label
								>
							</div>

							<label class="code-label"
								>Kaggle notebook cell
								<textarea readonly value={kaggleBootstrapCode()}></textarea>
							</label>

							<div class="remote-actions">
								<button class="ghost-button" type="button" onclick={openKaggleNotebook}
									>Open Notebook</button
								>
								<button class="ghost-button" type="button" onclick={copyKaggleBootstrap}
									>Copy Kaggle Cell</button
								>
								<button class="primary-action" type="button" onclick={connectKaggleTunnel}
									>Use Kaggle GPU</button
								>
							</div>
						</div>

						<div class="remote-list">
							<div class="remote-row">
								<span></span>
								<strong>{backendUrl || 'Backend'}</strong>
								<span>Status: {connectionState}</span>
								<small>WebSocket: /ws</small>
								<button type="button" aria-label="Reconnect backend" onclick={connectBackend}
									>sync</button
								>
							</div>
							<div class="remote-row">
								<span></span>
								<strong>YOLO</strong>
								<span>{modelReady ? 'Ready' : modelState}</span>
								<small>Variant: {modelVariant}</small>
								<button type="button" aria-label="Initialize YOLO model" onclick={initializeModel}
									>load</button
								>
							</div>
							{#if remotes.length}
								{#each remotes as model, index (`${model.name || model}-${index}`)}
									<div class="remote-row">
										<span></span>
										<strong>{model.name || model}</strong>
										<span>{model.status || 'Available'}</span>
										<small>{model.variant || model.type || 'Backend model'}</small>
										<button type="button" aria-label="Copy backend URL" onclick={copyCode}
											>copy</button
										>
									</div>
								{/each}
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
									<p>Upload video files<br />from local device</p>
									<span>MP4, MOV, WEBM, AVI, MKV</span>
									<input
										bind:this={localFileInput}
										class="hidden-file-input"
										type="file"
										accept="video/*,.mp4,.mov,.m4v,.webm,.avi,.mkv"
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
						<div class="split-grid">
							<div>
								<h2>Prompts</h2>
								<div class="form-group">
									<span class="field-label">Create new</span>
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
										accept=".zip,.yaml,.yml,.txt,.json,.csv,image/*,video/*,.mp4,.mov,.m4v,.webm,.avi,.mkv"
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

	.kaggle-card p {
		margin: 5px 0 0;
		max-width: 680px;
		color: #555;
		line-height: 1.55;
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

	.kaggle-steps {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 10px;
	}

	.kaggle-steps div {
		display: grid;
		grid-template-columns: 28px 1fr;
		align-items: start;
		gap: 10px;
		border-radius: 6px;
		background: #f5f5f5;
		padding: 10px;
	}

	.kaggle-steps strong {
		display: grid;
		place-items: center;
		width: 24px;
		height: 24px;
		border-radius: 50%;
		background: #d86f32;
		color: #fff;
		font-size: 0.72rem;
	}

	.kaggle-steps span {
		font-size: 0.78rem;
		line-height: 1.4;
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
