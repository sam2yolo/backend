// Decode SAM 3.1 masks (zlib+base64+packbits) and render result overlays.
import { inflate } from 'pako';

export const PALETTE = [
	[255, 59, 48], [52, 199, 89], [0, 122, 255], [255, 204, 0],
	[175, 82, 222], [255, 149, 0], [90, 200, 250], [255, 45, 85],
	[100, 210, 80], [191, 90, 242], [255, 105, 97], [48, 176, 199]
];

/** Decode one SAM mask object → { h, w, bits:Uint8Array(h*w) of 0/1 }. */
export function decodeMask(maskObj) {
	const shape = maskObj.shape || [];
	const binary = atob(maskObj.data);
	const packed = new Uint8Array(binary.length);
	for (let i = 0; i < binary.length; i++) packed[i] = binary.charCodeAt(i);
	const inflated = inflate(packed); // packbits bytes
	const total = shape.reduce((a, b) => a * b, 1);
	const bits = new Uint8Array(total);
	for (let i = 0; i < total; i++) {
		bits[i] = (inflated[i >> 3] >> (7 - (i & 7))) & 1;
	}
	const h = shape[shape.length - 2];
	const w = shape[shape.length - 1];
	return { h, w, bits };
}

/**
 * Render one image_result onto a canvas, optionally over a source image.
 * result = { width, height, boxes:[[x1,y1,x2,y2]], scores:[], masks:[{shape,data,encoding}] }
 */
export function renderResult(canvas, img, result, { alpha = 0.45 } = {}) {
	const w = result.width || (img && img.naturalWidth) || 640;
	const h = result.height || (img && img.naturalHeight) || 480;
	canvas.width = w;
	canvas.height = h;
	const ctx = canvas.getContext('2d');

	if (img) {
		ctx.drawImage(img, 0, 0, w, h);
	} else {
		ctx.fillStyle = '#0e1117';
		ctx.fillRect(0, 0, w, h);
	}

	const masks = result.masks || [];
	if (masks.length) {
		const frame = ctx.getImageData(0, 0, w, h);
		const data = frame.data;
		for (let mi = 0; mi < masks.length; mi++) {
			const color = PALETTE[mi % PALETTE.length];
			let decoded;
			try {
				decoded = decodeMask(masks[mi]);
			} catch {
				continue;
			}
			const bits = decoded.bits;
			// SAM masks come back at full image resolution (h*w === bits.length).
			const n = Math.min(bits.length, w * h);
			for (let i = 0; i < n; i++) {
				if (bits[i]) {
					const idx = i * 4;
					data[idx] = data[idx] * (1 - alpha) + color[0] * alpha;
					data[idx + 1] = data[idx + 1] * (1 - alpha) + color[1] * alpha;
					data[idx + 2] = data[idx + 2] * (1 - alpha) + color[2] * alpha;
				}
			}
		}
		ctx.putImageData(frame, 0, 0);
	}

	const boxes = result.boxes || [];
	const scores = result.scores || [];
	ctx.lineWidth = Math.max(2, w / 400);
	ctx.font = `${Math.max(12, Math.round(w / 55))}px ui-sans-serif, system-ui, sans-serif`;
	ctx.textBaseline = 'bottom';
	for (let i = 0; i < boxes.length; i++) {
		const c = PALETTE[i % PALETTE.length];
		const stroke = `rgb(${c[0]},${c[1]},${c[2]})`;
		const [x1, y1, x2, y2] = boxes[i];
		ctx.strokeStyle = stroke;
		ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
		if (scores[i] != null) {
			ctx.fillStyle = stroke;
			ctx.fillText(Number(scores[i]).toFixed(2), x1 + 2, Math.max(14, y1 - 2));
		}
	}
}
