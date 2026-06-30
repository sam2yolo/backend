const DB_NAME = 'samtoyolo-results';
const DB_VERSION = 1;
const STORE_NAME = 'frames';

let dbPromise;

function openDb() {
	if (typeof indexedDB === 'undefined') return Promise.reject(new Error('IndexedDB unavailable'));
	if (dbPromise) return dbPromise;

	dbPromise = new Promise((resolve, reject) => {
		const request = indexedDB.open(DB_NAME, DB_VERSION);
		request.onerror = () => reject(request.error || new Error('Could not open result store'));
		request.onupgradeneeded = () => {
			const db = request.result;
			if (!db.objectStoreNames.contains(STORE_NAME)) {
				const store = db.createObjectStore(STORE_NAME, { keyPath: 'key' });
				store.createIndex('taskId', 'taskId', { unique: false });
				store.createIndex('createdAt', 'createdAt', { unique: false });
			}
		};
		request.onsuccess = () => resolve(request.result);
	});

	return dbPromise;
}

function withStore(mode, callback) {
	return openDb().then(
		(db) =>
			new Promise((resolve, reject) => {
				const tx = db.transaction(STORE_NAME, mode);
				const store = tx.objectStore(STORE_NAME);
				const result = callback(store);
				tx.oncomplete = () => resolve(result);
				tx.onerror = () => reject(tx.error || new Error('Result store transaction failed'));
				tx.onabort = () => reject(tx.error || new Error('Result store transaction aborted'));
			})
	);
}

export function saveInferenceResult(record) {
	return withStore('readwrite', (store) => store.put(record));
}

export function deleteInferenceResult(key) {
	return withStore('readwrite', (store) => store.delete(key));
}

export function clearInferenceResults() {
	return withStore('readwrite', (store) => store.clear());
}

export function listInferenceResults(limit = 600) {
	return withStore(
		'readonly',
		(store) =>
			new Promise((resolve, reject) => {
				const items = [];
				const request = store.index('createdAt').openCursor(null, 'prev');
				request.onerror = () => reject(request.error || new Error('Could not read results'));
				request.onsuccess = () => {
					const cursor = request.result;
					if (!cursor || items.length >= limit) {
						resolve(items);
						return;
					}
					items.push(cursor.value);
					cursor.continue();
				};
			})
	);
}
