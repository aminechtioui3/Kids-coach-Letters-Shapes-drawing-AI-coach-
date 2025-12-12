// src/components/FingerDrawingCanvas.tsx
import React, { useEffect, useRef, useState } from 'react';
import { Hands, type Results } from '@mediapipe/hands';
import { Camera } from '@mediapipe/camera_utils';

export type TrajectoryPoint = { x: number; y: number; t: number };

interface FingerDrawingCanvasProps {
  onTrajectoryChange: (points: TrajectoryPoint[]) => void;
  startedAt: number; // timestamp ms
  resetKey: number;  // when this changes, canvas is cleared (new attempt)
}

const FingerDrawingCanvas: React.FC<FingerDrawingCanvasProps> = ({
  onTrajectoryChange,
  startedAt,
  resetKey,
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [points, setPoints] = useState<TrajectoryPoint[]>([]);
  const drawingActiveRef = useRef<boolean>(false);

  // Clear drawing when resetKey changes
  useEffect(() => {
    setPoints([]);
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }, [resetKey]);

  // Notify parent when points change
  useEffect(() => {
    onTrajectoryChange(points);
  }, [points, onTrajectoryChange]);

  // Init MediaPipe Hands + Camera once per session
  useEffect(() => {
    const videoEl = videoRef.current;
    const canvasEl = canvasRef.current;
    if (!videoEl || !canvasEl) return;

    let hands: Hands | null = null;
    let camera: Camera | null = null;
    let cancelled = false;

    const ctx = canvasEl.getContext('2d');
    if (!ctx) return;

    // Fill white background
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvasEl.width, canvasEl.height);

    const handleResults = (results: Results) => {
      if (cancelled) return;
      const landmarks = results.multiHandLandmarks?.[0];
      if (!landmarks) {
        drawingActiveRef.current = false;
        return;
      }

      const indexTip = landmarks[8]; // INDEX_FINGER_TIP
      const indexPip = landmarks[6]; // INDEX_FINGER_PIP

      // "Pen down": finger pointing up (tip above PIP)
      const isDrawing = indexTip.y < indexPip.y;
      drawingActiveRef.current = isDrawing;

      // Normalized coordinates from Mediapipe
      const xNormFlipped = 1 - indexTip.x; // flip horizontally for natural feel
      const yNorm = indexTip.y;

      const px = xNormFlipped * canvasEl.width;
      const py = yNorm * canvasEl.height;
      const t = (Date.now() - startedAt) / 1000;

      // 🔴 IMPORTANT CHANGE:
      // Only record points (and therefore only send them to backend)
      // when the "pen" is down. So the CNN sees only the strokes that
      // are actually drawn on the canvas.
      if (!isDrawing) {
        return;
      }

      setPoints((prev) => {
        const next: TrajectoryPoint[] = [
          ...prev,
          { x: xNormFlipped, y: yNorm, t },
        ];

        // Draw line from previous point
        if (prev.length > 0) {
          const last = prev[prev.length - 1];
          ctx.strokeStyle = '#000000';
          ctx.lineWidth = 4;
          ctx.lineCap = 'round';
          ctx.beginPath();
          ctx.moveTo(last.x * canvasEl.width, last.y * canvasEl.height);
          ctx.lineTo(px, py);
          ctx.stroke();
        } else {
          // First point: just put a small dot so the stroke starts visible
          ctx.fillStyle = '#000000';
          ctx.beginPath();
          ctx.arc(px, py, 2, 0, Math.PI * 2);
          ctx.fill();
        }

        return next;
      });
    };

    // Init Hands
    hands = new Hands({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });

    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.7,
    });

    hands.onResults(handleResults);

    // Init Camera
    camera = new Camera(videoEl, {
      onFrame: async () => {
        if (!hands || cancelled) return;
        try {
          await hands.send({ image: videoEl });
        } catch {
          // ignore errors after unmount
        }
      },
      width: 640,
      height: 480,
    });

    camera.start();

    return () => {
      cancelled = true;
      if (camera) {
        try {
          camera.stop();
        } catch {
          /* ignore */
        }
      }
      if (hands) {
        try {
          hands.close();
        } catch {
          /* ignore */
        }
      }
    };
  }, [startedAt]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{ fontSize: 14, fontWeight: 500 }}>
        Draw with your fingertip ✋
      </div>
      <div style={{ display: 'flex', gap: 12 }}>
        <video
          ref={videoRef}
          style={{
            width: 220,
            height: 160,
            borderRadius: 8,
            border: '1px solid #d1d5db',
            background: '#000',
            // Mirror webcam preview for natural interaction
            transform: 'scaleX(-1)',
          }}
          autoPlay
          muted
        />
        <canvas
          ref={canvasRef}
          width={500}
          height={300}
          style={{
            border: '2px solid #333',
            borderRadius: 8,
            background: '#ffffff',
          }}
        />
      </div>
      <small style={{ color: '#6b7280' }}>
        Tip: point your index finger up to “draw”; relax it to stop.
      </small>
    </div>
  );
};

export default FingerDrawingCanvas;
