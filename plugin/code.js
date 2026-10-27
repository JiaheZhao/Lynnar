"use strict";
// ---------------------------------------------
// code.ts  —  main thread of the Figma plugin
// ---------------------------------------------
figma.showUI(__html__, { width: 375, height: 560 });
const BACKEND_BASE_URL = 'http://localhost:5000';
// ---------- Helper ----------
function getSelectedFrame() {
    const node = figma.currentPage.selection[0];
    if (node && node.type === 'FRAME') {
        return node;
    }
    return null;
}
/**
 * Return an array of { id, name, x, y, width, height } for every direct child
 * of the given frame.  Coordinates are **relative to the frame’s top-left**.
 */
function getChildBBoxes(frame) {
    const fx = frame.absoluteTransform[0][2];
    const fy = frame.absoluteTransform[1][2];
    return frame.children.map(node => {
        const ax = node.absoluteTransform[0][2];
        const ay = node.absoluteTransform[1][2];
        return {
            id: node.id,
            name: node.name,
            x: ax - fx,
            y: ay - fy,
            width: node.width,
            height: node.height
        };
    });
}
function imageSizeForFrame(frame) {
    if (!frame || frame.height === 0)
        return '1024x1024';
    const ratio = frame.width / frame.height;
    if (ratio > 1.15)
        return '1536x1024';
    if (ratio < 0.87)
        return '1024x1536';
    return '1024x1024';
}
// ---------- Message handler ----------
figma.ui.onmessage = (msg) => {
    switch (msg.type) {
        /* 1. original rectangle demo --------------------------- */
        case 'create-shapes': {
            const nodes = [];
            for (let i = 0; i < msg.count; i++) {
                const rect = figma.createRectangle();
                rect.x = i * 150;
                rect.fills = [{ type: 'SOLID', color: { r: 1, g: 0.5, b: 0 } }];
                figma.currentPage.appendChild(rect);
                nodes.push(rect);
            }
            figma.currentPage.selection = nodes;
            figma.viewport.scrollAndZoomIntoView(nodes);
            figma.closePlugin(); // demo done
            break;
        }
        /* 2. NEW: return selected-frame size ------------------- */
        case 'send': {
            const selectedOptions = msg.selectedOptions || [];
            console.log('[CONTROLLER] Options selected by user:', selectedOptions);
            if (!selectedOptions || selectedOptions.length === 0) {
                figma.notify('Please select at least one option.');
                figma.ui.postMessage({ type: 'frame-size', error: 'no-options' });
                figma.ui.postMessage({ type: 'loading', loading: false }); // Reset UI
                return;
            }
            const frame = getSelectedFrame();
            if (!frame) {
                figma.notify('Please select a frame first');
                figma.ui.postMessage({ type: 'frame-size', error: 'no-frame' });
                figma.ui.postMessage({ type: 'loading', loading: false }); // Reset UI
                return;
            }
            const objects = getChildBBoxes(frame);
            console.log('[CONTROLLER] objects', objects);
            console.log(`[CONTROLLER] Frame size: ${frame.width} × ${frame.height}`);
            // Show loading notification with a long timeout so it stays visible
            figma.notify('Generating heatmaps...', { timeout: 10000 });
            figma.ui.postMessage({ type: 'loading', loading: true }); // Show spinner in UI
            // Send frame and child info to backend server for prediction
            fetch(`${BACKEND_BASE_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    frame: {
                        id: frame.id,
                        name: frame.name,
                        width: frame.width,
                        height: frame.height
                    },
                    children: objects,
                    selectedOptions
                })
            })
                .then(res => res.json())
                .then(async (result) => {
                // Helper to decode base64 to Uint8Array
                function base64ToBytes(b64) {
                    return figma.base64Decode(b64);
                }
                for (const [field, b64] of Object.entries(result)) {
                    const bytes = base64ToBytes(b64);
                    const image = figma.createImage(bytes);
                    // Insert as a rectangle in the same position/size as the frame
                    const rect = figma.createRectangle();
                    rect.resizeWithoutConstraints(frame.width, frame.height);
                    rect.x = frame.x;
                    rect.y = frame.y;
                    rect.fills = [{ type: 'IMAGE', scaleMode: 'FILL', imageHash: image.hash }];
                    rect.name = field; // Set the rectangle's name to the field name
                    rect.opacity = 0.5; // Set opacity to 50%
                    figma.currentPage.appendChild(rect);
                }
                figma.notify('Heatmap(s) inserted!');
                figma.ui.postMessage({ type: 'guides-ready' });
                figma.ui.postMessage({ type: 'loading', loading: false }); // Hide spinner
            })
                .catch(err => {
                console.error('Failed to fetch prediction or insert image:', err);
                figma.notify('Failed to fetch prediction or insert image.');
                figma.ui.postMessage({ type: 'guides-error' });
                figma.ui.postMessage({ type: 'loading', loading: false }); // Hide spinner
            });
            break;
        }
        /* 3. user clicked Cancel ------------------------------- */
        case 'cancel': {
            figma.closePlugin();
            break;
        }
        /* 4. Generate an image from the chat prompt ------------ */
        case 'generate-image': {
            const prompt = (msg.prompt || '').trim();
            if (!prompt) {
                figma.ui.postMessage({ type: 'image-generation-error', error: 'Describe a visual idea first.' });
                return;
            }
            const frame = getSelectedFrame();
            const size = imageSizeForFrame(frame);
            const placement = frame
                ? { width: frame.width, height: frame.height, x: frame.x + frame.width + 48, y: frame.y }
                : null;
            figma.ui.postMessage({ type: 'image-generation-start' });
            figma.notify('Generating image...', { timeout: 20000 });
            fetch(`${BACKEND_BASE_URL}/generate-image`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt,
                    provider: msg.provider || 'dalle',
                    size,
                    quality: 'medium'
                })
            })
                .then(async (res) => {
                let data;
                try {
                    data = await res.json();
                }
                catch (_a) {
                    throw new Error('The local server did not respond. Start it on port 5000, then try again.');
                }
                if (!res.ok || !data.success || !data.image_base64) {
                    throw new Error(data.error || data.message || 'Image generation failed');
                }
                return data.image_base64;
            })
                .then((b64) => {
                const image = figma.createImage(figma.base64Decode(b64));
                const rect = figma.createRectangle();
                rect.name = 'Lynnar image';
                if (placement) {
                    rect.resizeWithoutConstraints(placement.width, placement.height);
                    rect.x = placement.x;
                    rect.y = placement.y;
                }
                else {
                    rect.resize(512, 512);
                    const center = figma.viewport.center;
                    rect.x = center.x - 256;
                    rect.y = center.y - 256;
                }
                rect.fills = [{ type: 'IMAGE', scaleMode: 'FILL', imageHash: image.hash }];
                figma.currentPage.appendChild(rect);
                figma.currentPage.selection = [rect];
                figma.viewport.scrollAndZoomIntoView([rect]);
                figma.notify('Image placed on the canvas.');
                figma.ui.postMessage({ type: 'image-generation-success' });
            })
                .catch((err) => {
                console.error('Image generation failed:', err);
                figma.notify('Image generation failed.');
                figma.ui.postMessage({
                    type: 'image-generation-error',
                    error: err instanceof Error ? err.message : 'The local server did not respond.'
                });
            });
            break;
        }
    }
};
