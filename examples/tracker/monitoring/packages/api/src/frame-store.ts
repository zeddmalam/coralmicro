type PoseKeypoint = {
  name: string;
  x: number;
  y: number;
  score: number;
};

type Pose = {
  score: number;
  keypoints: PoseKeypoint[];
};

export type FramePayload = {
  poses?: Pose[];
  imageData?: string;
};

type FrameListener = (frame: FramePayload) => void;

let latestFrame: FramePayload = {
  imageData: "",
  poses: [],
};

const frameListeners = new Set<FrameListener>();

export function getLatestFramePayload(): FramePayload {
  return latestFrame;
}

export function setLatestFramePayload(frame: FramePayload): void {
  latestFrame = frame;
  for (const listener of frameListeners) {
    listener(frame);
  }
}

export function subscribeToFramePayload(listener: FrameListener): () => void {
  frameListeners.add(listener);
  return () => {
    frameListeners.delete(listener);
  };
}
