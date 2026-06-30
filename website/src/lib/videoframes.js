// Sample frames from a video (object URL) in the browser, for SAM (image-only).

const METADATA_TIMEOUT_MS = 15000;
const SEEK_TIMEOUT_MS = 15000;
const FRAME_CALLBACK_TIMEOUT_MS = 1000;

function videoError(video, fallback) {
	const error = video.error;
	if (!error) return new Error(fallback);
	return new Error(error.message || `${fallback} (${error.code})`);
}

function waitForVideoFrame(video) {
	if (typeof video.requestVideoFrameCallback !== 'function') {
		return new Promise((resolve) => requestAnimationFrame(resolve));
	}
	return new Promise((resolve) => {
		const timeout = setTimeout(resolve, FRAME_CALLBACK_TIMEOUT_MS);
		video.requestVideoFrameCallback(() => {
			clearTimeout(timeout);
			resolve();
		});
	});
}

function waitForFrameData(video) {
	if (video.readyState >= 2) return Promise.resolve();
	return new Promise((resolve, reject) => {
		const timeout = setTimeout(() => {
			cleanup();
			reject(new Error('Timed out while decoding video frame'));
		}, SEEK_TIMEOUT_MS);
		const cleanup = () => {
			clearTimeout(timeout);
			video.removeEventListener('loadeddata', onReady);
			video.removeEventListener('canplay', onReady);
			video.removeEventListener('error', onError);
		};
		const onReady = () => {
			cleanup();
			resolve();
		};
		const onError = () => {
			cleanup();
			reject(videoError(video, 'Could not decode video frame'));
		};
		video.addEventListener('loadeddata', onReady, { once: true });
		video.addEventListener('canplay', onReady, { once: true });
		video.addEventListener('error', onError, { once: true });
	});
}

async function waitForDrawableFrame(video) {
	await waitForFrameData(video);
	await waitForVideoFrame(video);
}

function waitForMetadata(video) {
	if (video.readyState >= 1) return Promise.resolve();
	return new Promise((resolve, reject) => {
		const timeout = setTimeout(() => {
			cleanup();
			reject(new Error('Timed out while reading video metadata'));
		}, METADATA_TIMEOUT_MS);
		const cleanup = () => {
			clearTimeout(timeout);
			video.removeEventListener('loadedmetadata', onLoaded);
			video.removeEventListener('error', onError);
		};
		const onLoaded = () => {
			cleanup();
			resolve();
		};
		const onError = () => {
			cleanup();
			reject(videoError(video, 'Could not load video'));
		};
		video.addEventListener('loadedmetadata', onLoaded, { once: true });
		video.addEventListener('error', onError, { once: true });
	});
}

async function seek(video, time) {
	const target = Math.min(time, Math.max(0, video.duration - 0.001));
	if (Number.isFinite(video.currentTime) && Math.abs(video.currentTime - target) < 0.001) {
		await waitForDrawableFrame(video);
		return;
	}

	await new Promise((resolve, reject) => {
		const timeout = setTimeout(() => {
			cleanup();
			reject(new Error(`Timed out while seeking video to ${target.toFixed(2)}s`));
		}, SEEK_TIMEOUT_MS);
		const cleanup = () => {
			clearTimeout(timeout);
			video.removeEventListener('seeked', onSeeked);
			video.removeEventListener('error', onError);
		};
		const onSeeked = () => {
			cleanup();
			resolve();
		};
		const onError = () => {
			cleanup();
			reject(videoError(video, 'Could not seek video'));
		};
		video.addEventListener('seeked', onSeeked, { once: true });
		video.addEventListener('error', onError, { once: true });
		try {
			video.currentTime = target;
		} catch (error) {
			cleanup();
			reject(error);
		}
	});
	await waitForDrawableFrame(video);
}

/**
 * Sample frames at `fps` frames/second. Returns [{ blob, time, index }].
 * onProgress(done, total, detail) is called as metadata is read, frames are
 * sought/encoded, and captures complete.
 */
export async function sampleVideoFrames(url, fps = 2, maxFrames = 0, onProgress) {
	const video = document.createElement('video');
	video.muted = true;
	video.playsInline = true;
	video.preload = 'auto';
	onProgress?.(0, 0, { phase: 'metadata' });
	const metadataReady = waitForMetadata(video);
	video.src = url;
	video.load();

	await metadataReady;

	const duration = video.duration || 0;
	const width = video.videoWidth;
	const height = video.videoHeight;
	if (!duration || !width || !height) throw new Error('Invalid video metadata');

	const step = 1 / Math.max(0.1, fps);
	const times = [];
	for (let t = 0; t < duration; t += step) {
		times.push(t);
		if (maxFrames && times.length >= maxFrames) break;
	}
	onProgress?.(0, times.length, { phase: 'planned', duration, width, height });

	const canvas = document.createElement('canvas');
	canvas.width = width;
	canvas.height = height;
	const ctx = canvas.getContext('2d');

	const frames = [];
	for (let i = 0; i < times.length; i++) {
		const detail = {
			index: i + 1,
			total: times.length,
			time: times[i],
			duration,
			width,
			height
		};
		onProgress?.(frames.length, times.length, { ...detail, phase: 'seeking' });
		await seek(video, times[i]);
		onProgress?.(frames.length, times.length, { ...detail, phase: 'encoding' });
		ctx.drawImage(video, 0, 0, width, height);
		const blob = await new Promise((resolve) =>
			canvas.toBlob((b) => resolve(b), 'image/jpeg', 0.9)
		);
		if (blob) frames.push({ blob, time: times[i], index: i });
		onProgress?.(frames.length, times.length, { ...detail, phase: 'captured' });
	}

	video.removeAttribute('src');
	video.load();
	return frames;
}
