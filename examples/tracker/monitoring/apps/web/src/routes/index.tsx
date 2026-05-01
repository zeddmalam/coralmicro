import { createFileRoute } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { useSubscription } from '@trpc/tanstack-react-query'
import { useTRPC } from '#/integrations/trpc/react'

export const Route = createFileRoute('/')({
  component: App,
  loader: async ({ context }) => {
    await context.queryClient.prefetchQuery(context.trpc.healthCheck.queryOptions())
  },
})

function App() {
  const trpc = useTRPC()
  const [latestPoseCount, setLatestPoseCount] = useState<number>(0)
  const [poseError, setPoseError] = useState<string | null>(null)
  const [frameSrc, setFrameSrc] = useState<string>('')
  const [imageSize, setImageSize] = useState<{ width: number; height: number } | null>(null)
  const describeFrame = useQuery({
    ...trpc.describeLatestFrame.queryOptions(),
    refetchInterval: 4000,
    enabled: Boolean(frameSrc),
  })
  const framePayload = useSubscription(
    trpc.framePayload.subscriptionOptions(undefined, {
      onData(data) {
        setLatestPoseCount(data.poses?.length ?? 0)
        setFrameSrc(data.imageData ? `data:image/jpeg;base64,${data.imageData}` : '')
        setPoseError(null)
      },
      onError(error) {
        setPoseError(error.message)
      },
    }),
  )

  return (
    <main className="page-wrap px-4 pb-8 pt-14">
      <section className="island-shell mt-8 rounded-2xl p-6">
        <p className="island-kicker mb-2">Frame stream</p>
        <div className="mb-6 rounded-xl border border-[rgba(50,143,151,0.24)] bg-white/70 p-4">
          <p className="mb-3 text-sm text-[var(--sea-ink-soft)]">
            Rendering <code>framePayload.imageData</code> from subscription
          </p>
          <div className="relative w-full max-w-xl">
            <img
              src={frameSrc}
              alt="Live frame"
              className="block w-full rounded-lg border border-[rgba(23,58,64,0.15)] bg-black/5"
              onLoad={(event) => {
                const target = event.currentTarget
                setImageSize({
                  width: target.naturalWidth || target.clientWidth || 1,
                  height: target.naturalHeight || target.clientHeight || 1,
                })
              }}
            />
            {describeFrame.data?.result.objects.map((object, index) => (
              <div
                key={`${object.name}-${index}`}
                className="absolute border-2 border-lime-400 bg-lime-300/10"
                style={{
                  left: `${((object.box.x ?? 0) / (imageSize?.width || 1)) * 100}%`,
                  top: `${((object.box.y ?? 0) / (imageSize?.height || 1)) * 100}%`,
                  width: `${((object.box.width ?? 0) / (imageSize?.width || 1)) * 100}%`,
                  height: `${((object.box.height ?? 0) / (imageSize?.height || 1)) * 100}%`,
                }}
              >
                <span className="absolute -top-6 left-0 rounded bg-lime-400 px-1.5 py-0.5 text-xs font-medium text-black">
                  {object.name}
                  {typeof object.confidence === 'number'
                    ? ` ${Math.round(object.confidence * 100)}%`
                    : ''}
                </span>
              </div>
            ))}
          </div>
        </div>

        <div className="mb-6 rounded-xl border border-[rgba(50,143,151,0.24)] bg-white/70 p-4">
          <p className="mb-3 text-sm text-[var(--sea-ink-soft)]">
            Subscribing to <code>trpc.framePayload</code> updates
          </p>
          <p className="text-sm text-[var(--sea-ink)]">
            {framePayload.status === 'connecting'
              ? 'Connecting pose stream...'
              : framePayload.status === 'error' || poseError
                ? `Pose stream failed: ${poseError ?? framePayload.error?.message ?? 'unknown error'}`
                : `Detected poses: ${latestPoseCount}`}
          </p>
          <p className="mt-1 text-sm text-[var(--sea-ink)]">
            {describeFrame.isError
              ? `Object detection failed: ${describeFrame.error.message}`
              : `Detected objects: ${describeFrame.data?.result.objects.length ?? 0}`}
          </p>
        </div>

      </section>
    </main>
  )
}
