const assert = require('node:assert/strict');

async function assertDirectorMusicLength(page) {
  const calls = [];
  await page.route('**/api/v1/director/music-clip-limits', route => route.fulfill({contentType: 'application/json', body: JSON.stringify({
    fps: 24, frames_minimum: 4, frame_step: 4, hard_max_frames: 344, recommended_frames: 240,
    max_frames: 240, max_seconds: 10, recommended_seconds: 10, auto: true,
  })}));
  await page.route('**/api/v1/studio-preferences', route => {
    if (route.request().method() === 'PUT') calls.push(route.request().postDataJSON());
    return route.fulfill({contentType: 'application/json', body: '{}'});
  });
  await page.evaluate(() => {
    window.store.setState({directorSkill: 'music_video', directorMusicClipSeconds: null, directorSongDuration: 30,
      directorSeamless: false, directorLoading: false, directorError: null, directorAnalysis: null});
    window.mountMusicLength();
  });
  const group = page.getByRole('group', {name: 'Music video clip length'});
  const auto = group.getByRole('button', {name: 'Auto', exact: true});
  const slider = group.getByRole('slider');
  await slider.waitFor();
  assert.equal(await auto.getAttribute('aria-pressed'), 'true');
  assert.equal(await slider.isEnabled(), true);
  await slider.focus(); await slider.press('Home'); await slider.press('ArrowRight');
  assert.equal(await auto.getAttribute('aria-pressed'), 'false');
  assert.equal(Number(await slider.inputValue()), 8);
  assert.equal(await page.evaluate(() => window.store.getState().directorMusicClipSeconds), 8 / 24);
  await auto.click();
  assert.equal(await auto.getAttribute('aria-pressed'), 'true');
  assert.equal(Number(await slider.inputValue()), 240);
  await page.evaluate(() => window.store.setState({directorAnalysis: {duration: 30},
    directorSetEnergyBias: async () => {window.replanCalled = true; window.store.setState({directorError: null});}}));
  await slider.press('ArrowLeft');
  await group.getByRole('button', {name: 'Update clip layout'}).click();
  assert.equal(await page.evaluate(() => window.replanCalled), true);
  await page.waitForTimeout(900);
  assert.ok(calls.some(call => typeof call.director_music_clip_seconds === 'number'), 'Explicit preference is saved');
  await page.evaluate(() => window.store.setState({directorSeamless: true}));
  assert.equal(await slider.isEnabled(), false);
}
module.exports = {assertDirectorMusicLength};
