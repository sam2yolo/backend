// Sample frames from a video (object URL) in the browser, for SAM (image-only).

function seek(video, time) {
	return new Promise((resolve, reject) => {
		const onSeeked = () => {
			video.removeEventListener('seeked', onSeeked);
			resolve();
		};
		video.addEventListener('seeked', onSeeked);
		video.addEventListener('error', reject, { once: true });
		video.currentTime = Math.min(time, Math.max(0, video.duration - 0.001));
	});
}

/**
 * Sample frames at `fps` frames/second. Returns [{ blob, time, index }].
 * onProgress(done, total) is called as frames are captured.
 */
export async function sampleVideoFrames(url, fps = 2, maxFrames = 0, onProgress) {
	const video = document.createElement('video');
	video.muted = true;
	video.playsInline = true;
	video.preload = 'auto';
	video.src = url;

	await new Promise((resolve, reject) => {
		video.addEventListener('loadedmetadata', resolve, { once: true });
		video.addEventListener('error', () => reject(new Error('Could not load video')), { once: true });
	});

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

	const canvas = document.createElement('canvas');
	canvas.width = width;
	canvas.height = height;
	const ctx = canvas.getContext('2d');

	const frames = [];
	for (let i = 0; i < times.length; i++) {
		await seek(video, times[i]);
		ctx.drawImage(video, 0, 0, width, height);
		const blob = await new Promise((resolve) =>
			canvas.toBlob((b) => resolve(b), 'image/jpeg', 0.9)
		);
		if (blob) frames.push({ blob, time: times[i], index: i });
		onProgress?.(frames.length, times.length);
	}

	video.removeAttribute('src');
	video.load();
	return frames;
}
