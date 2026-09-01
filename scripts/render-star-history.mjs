import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';

const DAY_MS = 24 * 60 * 60 * 1000;
const API_VERSION = '2022-11-28';

const palettes = {
  light: {
    background: '#ffffff', panel: '#f7fafc', grid: '#dbe4ea', text: '#17212b', muted: '#6a7b88',
    line: '#22a884', fillStart: '#22a88442', fillEnd: '#22a88405', dot: '#147d67',
  },
  dark: {
    background: '#0b1118', panel: '#101922', grid: '#263443', text: '#e7eef5', muted: '#8292a2',
    line: '#53e1bf', fillStart: '#53e1bf4a', fillEnd: '#53e1bf08', dot: '#8df1d8',
  },
};

function escapeXml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;',
  })[character]);
}

function startOfUtcDay(value) {
  const date = new Date(value);
  return Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate());
}

function dateLabel(timestamp) {
  return new Intl.DateTimeFormat('en', { month: 'short', day: 'numeric', year: '2-digit', timeZone: 'UTC' }).format(timestamp);
}

export function normalizeStarHistory({ repository, created_at: createdAt, starred_at: starredAt, updated_at: updatedAt = new Date().toISOString() }) {
  if (!repository || !createdAt || !Array.isArray(starredAt)) throw new Error('Star history requires repository, created_at, and starred_at');
  const stars = starredAt.map((value) => new Date(value).getTime()).filter(Number.isFinite).sort((left, right) => left - right);
  return { repository, createdAt: new Date(createdAt).toISOString(), stars, updatedAt: new Date(updatedAt).toISOString() };
}

export function renderStarHistory(input, theme = 'light') {
  const data = normalizeStarHistory(input);
  const colors = palettes[theme] ?? palettes.light;
  const width = 960;
  const height = 360;
  const plot = { left: 72, top: 90, right: 924, bottom: 260 };
  const created = startOfUtcDay(data.createdAt);
  const latest = data.stars.length ? startOfUtcDay(data.stars[data.stars.length - 1]) : created;
  const updated = startOfUtcDay(data.updatedAt);
  const end = Math.max(created + 7 * DAY_MS, latest, updated);
  const span = Math.max(DAY_MS, end - created);
  const maximum = Math.max(4, Math.ceil(data.stars.length / 4) * 4);
  const x = (timestamp) => plot.left + (Math.min(end, Math.max(created, timestamp)) - created) / span * (plot.right - plot.left);
  const y = (count) => plot.bottom - count / maximum * (plot.bottom - plot.top);

  const lineParts = [`M ${x(created).toFixed(2)} ${y(0).toFixed(2)}`];
  data.stars.forEach((timestamp, index) => {
    lineParts.push(`H ${x(timestamp).toFixed(2)} V ${y(index + 1).toFixed(2)}`);
  });
  lineParts.push(`H ${x(end).toFixed(2)}`);
  const linePath = lineParts.join(' ');
  const areaPath = `${linePath} L ${x(end).toFixed(2)} ${plot.bottom} L ${x(created).toFixed(2)} ${plot.bottom} Z`;

  const yGrid = Array.from({ length: 5 }, (_, index) => {
    const count = maximum * index / 4;
    const position = y(count);
    return `<g><line x1="${plot.left}" x2="${plot.right}" y1="${position}" y2="${position}"/><text x="${plot.left - 14}" y="${position + 4}" text-anchor="end">${Math.round(count)}</text></g>`;
  }).join('');
  const xGrid = Array.from({ length: 5 }, (_, index) => {
    const timestamp = created + span * index / 4;
    const position = x(timestamp);
    return `<g><line x1="${position}" x2="${position}" y1="${plot.top}" y2="${plot.bottom}"/><text x="${position}" y="${plot.bottom + 25}" text-anchor="middle">${escapeXml(dateLabel(timestamp))}</text></g>`;
  }).join('');
  const starLabel = `${data.stars.length} ${data.stars.length === 1 ? 'star' : 'stars'}`;
  const gradientId = `star-fill-${theme}`;

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="title description">
  <title id="title">${escapeXml(data.repository)} star history</title>
  <desc id="description">${escapeXml(starLabel)} as of ${escapeXml(data.updatedAt.slice(0, 10))}</desc>
  <defs>
    <linearGradient id="${gradientId}" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="${colors.fillStart}"/><stop offset="1" stop-color="${colors.fillEnd}"/>
    </linearGradient>
  </defs>
  <rect width="960" height="360" rx="18" fill="${colors.background}"/>
  <rect x="18" y="18" width="924" height="324" rx="14" fill="${colors.panel}" stroke="${colors.grid}"/>
  <text x="54" y="57" fill="${colors.text}" font-family="ui-sans-serif,system-ui,-apple-system,sans-serif" font-size="22" font-weight="700">${escapeXml(data.repository)} Star History</text>
  <text x="906" y="57" fill="${colors.line}" text-anchor="end" font-family="ui-sans-serif,system-ui,-apple-system,sans-serif" font-size="20" font-weight="700">${escapeXml(starLabel)}</text>
  <g fill="${colors.muted}" stroke="${colors.grid}" stroke-width="1" font-family="ui-sans-serif,system-ui,-apple-system,sans-serif" font-size="11">${yGrid}${xGrid}</g>
  <path d="${areaPath}" fill="url(#${gradientId})"/>
  <path d="${linePath}" fill="none" stroke="${colors.line}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>
  <circle cx="${x(end).toFixed(2)}" cy="${y(data.stars.length).toFixed(2)}" r="5" fill="${colors.dot}" stroke="${colors.background}" stroke-width="2"/>
  <text x="54" y="326" fill="${colors.muted}" font-family="ui-sans-serif,system-ui,-apple-system,sans-serif" font-size="11">Updated ${escapeXml(data.updatedAt.slice(0, 10))} · Source: GitHub API</text>
</svg>`;
}

async function githubJson(url, token, accept = 'application/vnd.github+json') {
  const response = await fetch(url, { headers: {
    Accept: accept,
    Authorization: `Bearer ${token}`,
    'X-GitHub-Api-Version': API_VERSION,
    'User-Agent': 'LabelOne-Star-History',
  } });
  if (!response.ok) throw new Error(`GitHub API ${response.status}: ${await response.text()}`);
  return response.json();
}

async function fetchStarHistory(repository, token) {
  const api = process.env.GITHUB_API_URL || 'https://api.github.com';
  const metadata = await githubJson(`${api}/repos/${repository}`, token);
  const starredAt = [];
  for (let page = 1; ; page += 1) {
    const rows = await githubJson(
      `${api}/repos/${repository}/stargazers?per_page=100&page=${page}`,
      token,
      'application/vnd.github.star+json',
    );
    starredAt.push(...rows.map((row) => row.starred_at).filter(Boolean));
    if (rows.length < 100) break;
  }
  return { repository, created_at: metadata.created_at, starred_at: starredAt, updated_at: new Date().toISOString() };
}

async function main() {
  const outputDirectory = process.argv[2] || 'docs/assets';
  const fixturePath = process.env.STAR_HISTORY_FIXTURE;
  const data = fixturePath
    ? JSON.parse(await readFile(fixturePath, 'utf8'))
    : await fetchStarHistory(process.env.GITHUB_REPOSITORY, process.env.GITHUB_TOKEN);
  await mkdir(outputDirectory, { recursive: true });
  await Promise.all([
    writeFile(`${outputDirectory}/star-history.svg`, renderStarHistory(data, 'light')),
    writeFile(`${outputDirectory}/star-history-dark.svg`, renderStarHistory(data, 'dark')),
  ]);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => { console.error(error); process.exitCode = 1; });
}
