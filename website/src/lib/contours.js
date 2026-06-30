// Dependency-free mask → polygon extraction for YOLO-segmentation export.
//
// Given a 2D uint8 bitmap (0/1) the size of width*height, find connected
// components (8-connectivity) and trace each component's outer boundary with a
// Moore-neighbour boundary follower, then simplify with Douglas–Peucker.
// Returns polygons in PIXEL coordinates: [{ points: [[x,y], ...] }].

/** Perpendicular-distance Douglas–Peucker simplification of a point list. */
function douglasPeucker(points, tol) {
	if (points.length < 3) return points.slice();
	const keep = new Uint8Array(points.length);
	keep[0] = 1;
	keep[points.length - 1] = 1;
	const stack = [[0, points.length - 1]];
	const tol2 = tol * tol;

	while (stack.length) {
		const [start, end] = stack.pop();
		const [ax, ay] = points[start];
		const [bx, by] = points[end];
		const dx = bx - ax;
		const dy = by - ay;
		const segLen2 = dx * dx + dy * dy;
		let maxDist2 = -1;
		let maxIdx = -1;
		for (let i = start + 1; i < end; i++) {
			const [px, py] = points[i];
			let dist2;
			if (segLen2 === 0) {
				const ex = px - ax;
				const ey = py - ay;
				dist2 = ex * ex + ey * ey;
			} else {
				// Distance from point to segment (a,b).
				const t = ((px - ax) * dx + (py - ay) * dy) / segLen2;
				const tc = t < 0 ? 0 : t > 1 ? 1 : t;
				const cx = ax + tc * dx;
				const cy = ay + tc * dy;
				const ex = px - cx;
				const ey = py - cy;
				dist2 = ex * ex + ey * ey;
			}
			if (dist2 > maxDist2) {
				maxDist2 = dist2;
				maxIdx = i;
			}
		}
		if (maxDist2 > tol2 && maxIdx > 0) {
			keep[maxIdx] = 1;
			stack.push([start, maxIdx], [maxIdx, end]);
		}
	}

	const out = [];
	for (let i = 0; i < points.length; i++) if (keep[i]) out.push(points[i]);
	return out;
}

// Moore-neighbour offsets, clockwise starting from the left of travel.
const MOORE = [
	[-1, 0],
	[-1, -1],
	[0, -1],
	[1, -1],
	[1, 0],
	[1, 1],
	[0, 1],
	[-1, 1]
];

/**
 * Trace the outer boundary of one connected component starting at a known
 * boundary pixel, using Moore-neighbour boundary following (Radial sweep).
 * `inside(x,y)` returns true if a pixel belongs to the component's label.
 */
function traceContour(startX, startY, inside, width, height) {
	const contour = [[startX, startY]];
	let cx = startX;
	let cy = startY;
	// Direction we came from (backtrack); start by pretending we entered from the left.
	let backDir = 0;
	const maxSteps = width * height * 4;
	let steps = 0;

	while (steps++ < maxSteps) {
		let found = false;
		// Scan the 8 neighbours clockwise starting just after the backtrack dir.
		for (let i = 0; i < 8; i++) {
			const dir = (backDir + 1 + i) % 8;
			const nx = cx + MOORE[dir][0];
			const ny = cy + MOORE[dir][1];
			if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
			if (inside(nx, ny)) {
				// Backtrack direction is opposite of the move we just made.
				backDir = (dir + 4) % 8;
				cx = nx;
				cy = ny;
				found = true;
				break;
			}
		}
		if (!found) break; // isolated pixel
		if (cx === startX && cy === startY) break; // closed loop
		contour.push([cx, cy]);
	}
	return contour;
}

/**
 * @param {Uint8Array} mask  width*height bitmap of 0/1 (row-major)
 * @param {number} width
 * @param {number} height
 * @param {{minArea?:number, simplifyTol?:number}} [opts]
 * @returns {Array<{points:[number,number][]}>}
 */
export function maskToPolygons(mask, width, height, opts = {}) {
	const { minArea = 16, simplifyTol = 1.5 } = opts;
	const labels = new Int32Array(width * height); // 0 = unvisited/background
	const polys = [];
	let nextLabel = 0;

	const at = (x, y) => mask[y * width + x] === 1;

	// Flood-fill each unlabeled foreground component (8-connectivity), counting
	// area, then trace its boundary from the component's top-left-most pixel.
	const stack = new Int32Array(width * height);
	for (let sy = 0; sy < height; sy++) {
		for (let sx = 0; sx < width; sx++) {
			const idx0 = sy * width + sx;
			if (mask[idx0] !== 1 || labels[idx0] !== 0) continue;
			nextLabel += 1;
			const label = nextLabel;
			let sp = 0;
			stack[sp++] = idx0;
			labels[idx0] = label;
			let area = 0;
			let startX = sx;
			let startY = sy;
			while (sp > 0) {
				const idx = stack[--sp];
				const x = idx % width;
				const y = (idx - x) / width;
				area += 1;
				// Track top-most then left-most pixel as a stable trace start.
				if (y < startY || (y === startY && x < startX)) {
					startX = x;
					startY = y;
				}
				for (let dy = -1; dy <= 1; dy++) {
					for (let dx = -1; dx <= 1; dx++) {
						if (dx === 0 && dy === 0) continue;
						const nx = x + dx;
						const ny = y + dy;
						if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
						const nIdx = ny * width + nx;
						if (mask[nIdx] === 1 && labels[nIdx] === 0) {
							labels[nIdx] = label;
							stack[sp++] = nIdx;
						}
					}
				}
			}
			if (area < minArea) continue;

			const inside = (x, y) => labels[y * width + x] === label;
			let contour = traceContour(startX, startY, inside, width, height);
			contour = douglasPeucker(contour, simplifyTol);
			if (contour.length >= 3) polys.push({ points: contour });
		}
	}
	return polys;
}

// --- Tiny self-check (run with: node src/lib/contours.js) ------------------
if (typeof process !== 'undefined' && process.argv?.[1]?.endsWith('contours.js')) {
	const W = 40;
	const H = 30;
	const mask = new Uint8Array(W * H);
	// Filled rectangle from (8,5) to (30,20) inclusive.
	const rx1 = 8;
	const ry1 = 5;
	const rx2 = 30;
	const ry2 = 20;
	for (let y = ry1; y <= ry2; y++) {
		for (let x = rx1; x <= rx2; x++) mask[y * W + x] = 1;
	}
	const polys = maskToPolygons(mask, W, H, { minArea: 4, simplifyTol: 1.5 });
	let pass = polys.length === 1;
	let bbox = null;
	if (pass) {
		const pts = polys[0].points;
		pass = pass && pts.length >= 4 && pts.length <= 8; // ~4 corners
		let minX = Infinity,
			minY = Infinity,
			maxX = -Infinity,
			maxY = -Infinity;
		for (const [x, y] of pts) {
			if (x < minX) minX = x;
			if (y < minY) minY = y;
			if (x > maxX) maxX = x;
			if (y > maxY) maxY = y;
		}
		bbox = { minX, minY, maxX, maxY };
		const tol = 1.5;
		pass =
			pass &&
			Math.abs(minX - rx1) <= tol &&
			Math.abs(minY - ry1) <= tol &&
			Math.abs(maxX - rx2) <= tol &&
			Math.abs(maxY - ry2) <= tol;
	}
	console.log(
		`${pass ? 'PASS' : 'FAIL'} — polys=${polys.length} points=${polys[0]?.points.length} bbox=${JSON.stringify(bbox)} expected=${JSON.stringify({ minX: rx1, minY: ry1, maxX: rx2, maxY: ry2 })}`
	);
	if (!pass) process.exitCode = 1;
}
