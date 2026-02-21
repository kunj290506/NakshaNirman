import { useMemo } from 'react'

/**
 * BoundaryPreview — SVG overlay of the original DXF boundary and the
 * buildable (setback-adjusted) polygon.
 *
 * Props:
 *   boundaryData.boundary       — original polygon coords [[x,y], ...]
 *   boundaryData.usable_polygon — setback-adjusted polygon coords
 *   boundaryData.area           — total plot area (sq.m)
 *   boundaryData.usable_area    — buildable area (sq.m)
 *   boundaryData.setback        — setback distance (m)
 *   boundaryData.coverage_ratio — usable / total ratio
 */
export default function BoundaryPreview({ boundaryData }) {
    const svgData = useMemo(() => {
        if (!boundaryData?.boundary || !boundaryData?.usable_polygon) return null

        const boundary = boundaryData.boundary
        const usable = boundaryData.usable_polygon

        // Compute bounding box over ALL points
        const allPts = [...boundary, ...usable]
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
        for (const [x, y] of allPts) {
            if (x < minX) minX = x
            if (y < minY) minY = y
            if (x > maxX) maxX = x
            if (y > maxY) maxY = y
        }

        const W = maxX - minX || 1
        const H = maxY - minY || 1
        const pad = Math.max(W, H) * 0.12

        const vbX = minX - pad
        const vbY = minY - pad
        const vbW = W + pad * 2
        const vbH = H + pad * 2

        // Build SVG path strings (flip Y so +Y is up)
        const toPath = (coords) =>
            coords
                .map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0]},${-(p[1])}`)
                .join(' ') + ' Z'

        const boundaryPath = toPath(boundary)
        const usablePath = toPath(usable)

        // Flip viewBox Y
        const flipVbY = -(minY + H + pad)

        // Compute label positions (centroid of original boundary)
        const cx = boundary.reduce((s, p) => s + p[0], 0) / boundary.length
        const cy = -(boundary.reduce((s, p) => s + p[1], 0) / boundary.length)

        // Stroke width relative to plot size
        const strokeW = Math.max(W, H) * 0.008

        return { boundaryPath, usablePath, vbX, flipVbY, vbW, vbH, cx, cy, strokeW, W, H }
    }, [boundaryData])

    if (!svgData) {
        return (
            <div className="preview-empty">
                <div className="preview-empty-icon">
                    <svg width="32" height="32" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7" />
                    </svg>
                </div>
                <h3>No Boundary Uploaded</h3>
                <p>Upload a DXF file to see the boundary preview with setback overlay</p>
            </div>
        )
    }

    const { boundaryPath, usablePath, vbX, flipVbY, vbW, vbH, cx, cy, strokeW } = svgData
    const dashLen = strokeW * 4

    return (
        <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* Legend */}
            <div style={{
                display: 'flex', justifyContent: 'center', gap: '1.5rem',
                padding: '0.6rem 1rem',
                background: 'var(--bg-secondary, #fafafa)',
                borderBottom: '1px solid var(--border)',
                fontSize: '0.78rem', fontWeight: 500,
            }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <span style={{
                        display: 'inline-block', width: 20, height: 3,
                        background: '#2563eb', borderRadius: 2,
                    }}></span>
                    Plot Boundary ({boundaryData.area?.toFixed(1)} sq.m)
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <span style={{
                        display: 'inline-block', width: 20, height: 3, borderRadius: 2,
                        background: 'repeating-linear-gradient(90deg, #16a34a 0, #16a34a 4px, transparent 4px, transparent 7px)',
                    }}></span>
                    Buildable Area ({boundaryData.usable_area?.toFixed(1)} sq.m)
                </span>
                <span style={{ color: 'var(--text-muted)' }}>
                    Setback: {boundaryData.setback}m
                </span>
            </div>

            {/* SVG viewport */}
            <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
                <svg
                    viewBox={`${vbX} ${flipVbY} ${vbW} ${vbH}`}
                    style={{ width: '100%', height: '100%' }}
                    preserveAspectRatio="xMidYMid meet"
                >
                    {/* Grid pattern */}
                    <defs>
                        <pattern id="grid" width={vbW / 10} height={vbH / 10} patternUnits="userSpaceOnUse">
                            <path d={`M ${vbW / 10} 0 L 0 0 0 ${vbH / 10}`}
                                fill="none" stroke="#e5e7eb" strokeWidth={strokeW * 0.3} />
                        </pattern>
                    </defs>
                    <rect x={vbX} y={flipVbY} width={vbW} height={vbH} fill="url(#grid)" />

                    {/* Original boundary — solid blue fill + stroke */}
                    <path
                        d={boundaryPath}
                        fill="rgba(37, 99, 235, 0.08)"
                        stroke="#2563eb"
                        strokeWidth={strokeW * 1.5}
                        strokeLinejoin="round"
                    />

                    {/* Usable (setback) polygon — dashed green fill + stroke */}
                    <path
                        d={usablePath}
                        fill="rgba(22, 163, 74, 0.12)"
                        stroke="#16a34a"
                        strokeWidth={strokeW * 1.5}
                        strokeDasharray={`${dashLen} ${dashLen * 0.6}`}
                        strokeLinejoin="round"
                    />

                    {/* Setback hatching between the two polygons */}
                    <defs>
                        <pattern id="setback-hatch" width={strokeW * 6} height={strokeW * 6} patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
                            <line x1="0" y1="0" x2="0" y2={strokeW * 6} stroke="rgba(239, 68, 68, 0.15)" strokeWidth={strokeW} />
                        </pattern>
                        <clipPath id="setback-clip">
                            <path d={boundaryPath} />
                        </clipPath>
                    </defs>

                    {/* Label: "Setback Zone" between boundaries — subtle red hatch area */}

                    {/* Vertex dots on boundary */}
                    {boundaryData.boundary.slice(0, -1).map(([x, y], i) => (
                        <circle key={`bv-${i}`} cx={x} cy={-y} r={strokeW * 2.5}
                            fill="#2563eb" stroke="white" strokeWidth={strokeW * 0.7} />
                    ))}

                    {/* Vertex dots on usable polygon */}
                    {boundaryData.usable_polygon.slice(0, -1).map(([x, y], i) => (
                        <circle key={`uv-${i}`} cx={x} cy={-y} r={strokeW * 2}
                            fill="#16a34a" stroke="white" strokeWidth={strokeW * 0.5} />
                    ))}

                    {/* Dimension labels on boundary edges */}
                    {boundaryData.boundary.slice(0, -1).map(([x1, y1], i) => {
                        const [x2, y2] = boundaryData.boundary[(i + 1) % (boundaryData.boundary.length - 1)]
                        const mx = (x1 + x2) / 2
                        const my = -((y1 + y2) / 2)
                        const len = Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        // Offset label outward from centroid
                        const dx = x2 - x1, dy = y2 - y1
                        const nx = -dy, ny = dx  // normal
                        const nl = Math.sqrt(nx * nx + ny * ny) || 1
                        const offset = strokeW * 6
                        // Determine sign: push outward from centroid
                        const centX = boundaryData.boundary.reduce((s, p) => s + p[0], 0) / (boundaryData.boundary.length - 1)
                        const centY = -(boundaryData.boundary.reduce((s, p) => s + p[1], 0) / (boundaryData.boundary.length - 1))
                        const dirX = mx - centX, dirY = my - centY
                        const dot = dirX * (nx / nl) + dirY * (ny / nl)
                        const sign = dot >= 0 ? 1 : -1
                        return (
                            <text
                                key={`dim-${i}`}
                                x={mx + sign * (nx / nl) * offset}
                                y={my + sign * (ny / nl) * offset}
                                textAnchor="middle"
                                dominantBaseline="middle"
                                fill="#374151"
                                fontSize={strokeW * 5}
                                fontWeight="600"
                                fontFamily="system-ui, sans-serif"
                            >
                                {len.toFixed(1)}m
                            </text>
                        )
                    })}
                </svg>
            </div>

            {/* Stats bar */}
            <div style={{
                display: 'flex', justifyContent: 'space-around',
                padding: '0.6rem',
                background: 'var(--bg-secondary, #fafafa)',
                borderTop: '1px solid var(--border)',
                fontSize: '0.75rem',
            }}>
                <div style={{ textAlign: 'center' }}>
                    <div style={{ fontWeight: 700, fontSize: '1rem', color: '#2563eb' }}>
                        {boundaryData.area?.toFixed(1)}
                    </div>
                    <div style={{ color: 'var(--text-muted)' }}>Total (sq.m)</div>
                </div>
                <div style={{ textAlign: 'center' }}>
                    <div style={{ fontWeight: 700, fontSize: '1rem', color: '#16a34a' }}>
                        {boundaryData.usable_area?.toFixed(1)}
                    </div>
                    <div style={{ color: 'var(--text-muted)' }}>Usable (sq.m)</div>
                </div>
                <div style={{ textAlign: 'center' }}>
                    <div style={{ fontWeight: 700, fontSize: '1rem', color: '#dc2626' }}>
                        {(boundaryData.area - boundaryData.usable_area)?.toFixed(1)}
                    </div>
                    <div style={{ color: 'var(--text-muted)' }}>Setback Loss</div>
                </div>
                <div style={{ textAlign: 'center' }}>
                    <div style={{ fontWeight: 700, fontSize: '1rem', color: 'var(--accent)' }}>
                        {(boundaryData.coverage_ratio * 100).toFixed(1)}%
                    </div>
                    <div style={{ color: 'var(--text-muted)' }}>Coverage</div>
                </div>
            </div>
        </div>
    )
}
