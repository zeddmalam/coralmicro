import { createFileRoute } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { useTRPC } from '#/integrations/trpc/react'

export const Route = createFileRoute('/demo/trpc-todo')({
  component: TRPCTodos,
  loader: async ({ context }) => {
    await context.queryClient.prefetchQuery(context.trpc.healthCheck.queryOptions())
  },
})

function TRPCTodos() {
  const trpc = useTRPC()
  const healthCheck = useQuery(trpc.healthCheck.queryOptions())

  return (
    <div
      className="flex items-center justify-center min-h-screen bg-gradient-to-br from-purple-100 to-blue-100 p-4 text-white"
      style={{
        backgroundImage:
          'radial-gradient(50% 50% at 95% 5%, #4a90c2 0%, #317eb9 50%, #1e4d72 100%)',
      }}
    >
      <div className="w-full max-w-2xl p-8 rounded-xl backdrop-blur-md bg-black/50 shadow-xl border-8 border-black/10">
        <h1 className="text-2xl mb-4">tRPC API connectivity</h1>
        <div className="space-y-2 rounded-lg border border-white/20 bg-white/10 p-4">
          <div className="text-white/70">Query: <code>healthCheck</code></div>
          <div className="text-lg">
            {healthCheck.isLoading ? 'Loading...' : `Result: ${healthCheck.data ?? 'Unavailable'}`}
          </div>
        </div>
      </div>
    </div>
  )
}
