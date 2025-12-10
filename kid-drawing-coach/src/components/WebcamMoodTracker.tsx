// src/components/WebcamMoodTracker.tsx
import React, {
  useEffect,
  useRef,
  useState,
  useCallback,
} from 'react';

type WebcamMoodTrackerProps = {
  active: boolean; // true while child is drawing
  onSnapshotsChange: (snapshots: string[]) => void;
  captureIntervalMs?: number;
  maxSnapshots?: number;
};

const WebcamMoodTracker: React.FC<WebcamMoodTrackerProps> = ({
  active,
  onSnapshotsChange,
  captureIntervalMs = 1500,
  maxSnapshots = 10,
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const captureTimerRef = useRef<number | null>(null);
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);

  const stopStream = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
  }, []);

  const clearTimer = useCallback(() => {
    if (captureTimerRef.current !== null) {
      window.clearInterval(captureTimerRef.current);
      captureTimerRef.current = null;
    }
  }, []);

  // Ask for camera access on mount
  useEffect(() => {
    let isMounted = true;

    const setupCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user' },
          audio: false,
        });
        if (!isMounted) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        console.log('[WebcamMoodTracker] got camera stream');
        streamRef.current = stream;
        setHasPermission(true);

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current
            .play()
            .catch(() => {
              // ignore autoplay error
            });
        }
      } catch (err) {
        console.error('[WebcamMoodTracker] getUserMedia error:', err);
        setHasPermission(false);
      }
    };

    setupCamera();

    return () => {
      isMounted = false;
      clearTimer();
      stopStream();
    };
  }, [clearTimer, stopStream]);

  // Capture frames when active = true
  useEffect(() => {
    if (!active) {
      clearTimer();
      return;
    }
    if (hasPermission === false) return;

    let snapshots: string[] = [];

    const captureAndUpdate = () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas) return;

      const width = video.videoWidth || 640;
      const height = video.videoHeight || 480;
      canvas.width = width;
      canvas.height = height;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      ctx.drawImage(video, 0, 0, width, height);

      // JPEG compression to reduce size
      const dataUrl = canvas.toDataURL('image/jpeg', 0.5);

      snapshots = [...snapshots, dataUrl];
      if (snapshots.length > maxSnapshots) {
        snapshots = snapshots.slice(-maxSnapshots);
      }
      console.log('[WebcamMoodTracker] snapshots count:', snapshots.length);
      onSnapshotsChange(snapshots);
    };

    // immediate capture once, then interval
    captureAndUpdate();
    const timerId = window.setInterval(captureAndUpdate, captureIntervalMs);
    captureTimerRef.current = timerId as unknown as number;

    return () => {
      clearTimer();
    };
  }, [active, hasPermission, captureIntervalMs, maxSnapshots, clearTimer, onSnapshotsChange]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{
          width: 200,
          height: 150,
          borderRadius: 12,
          objectFit: 'cover',
          background: '#000',
        }}
      />
      {hasPermission === false && (
        <p style={{ fontSize: 12, color: 'red' }}>
          Camera permission denied. Mood tracking will be disabled.
        </p>
      )}
      {/* Hidden canvas just for capturing frames */}
      <canvas ref={canvasRef} style={{ display: 'none' }} />
    </div>
  );
};

export default WebcamMoodTracker;
