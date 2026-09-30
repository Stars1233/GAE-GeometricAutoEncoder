// Frontend assets are independently reloadable without restarting the GPU engine.
(async () => {
    const base = '/gradio_api/file=__ASSET_ROOT__';
    const responses = await Promise.all(['camera_studio.css', 'camera_editor.js'].map(name =>
        fetch(`${base}/${name}?v=${Date.now()}`, {cache:'no-store'})));
    if (responses.some(response => !response.ok)) throw new Error('Could not load the camera editor. Reload this page.');
    const [css, source] = await Promise.all(responses.map(response => response.text()));
    let style = document.getElementById('gae-camera-studio-style');
    if (!style) { style = document.createElement('style'); style.id = 'gae-camera-studio-style'; document.head.appendChild(style); }
    style.textContent = css;
    new Function('element', 'props', 'watch', source)(element, props, watch);
})().catch(error => { element.textContent = error.message; });
