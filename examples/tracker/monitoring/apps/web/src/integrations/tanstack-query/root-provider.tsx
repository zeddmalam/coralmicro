import type { ReactNode } from 'react'
import { QueryClient } from '@tanstack/react-query'
import superjson from 'superjson'
import {
  createTRPCClient,
  httpBatchStreamLink,
  httpSubscriptionLink,
  splitLink,
} from '@trpc/client'
import { createTRPCOptionsProxy } from '@trpc/tanstack-react-query'

import type { AppRouter } from '@monitoring/api/routers/index'
import { TRPCProvider } from '#/integrations/trpc/react'

function getUrl() {
  const configuredBase = process.env.VITE_API_URL || 'http://localhost:3002'
  const normalizedBase = configuredBase.replace(/\/$/, '')
  return `${normalizedBase}/api/trpc`
}

export const trpcClient = createTRPCClient<AppRouter>({
  links: [
    splitLink({
      condition(op) {
        return op.type === 'subscription'
      },
      true: httpSubscriptionLink({
        url: getUrl(),
      }),
      false: httpBatchStreamLink({
        url: getUrl(),
      }),
    }),
  ],
})

export function getContext() {
  const queryClient = new QueryClient({
    defaultOptions: {
      dehydrate: { serializeData: superjson.serialize },
      hydrate: { deserializeData: superjson.deserialize },
    },
  })

  const serverHelpers = createTRPCOptionsProxy({
    client: trpcClient,
    queryClient: queryClient,
  })
  const context = {
    queryClient,
    trpc: serverHelpers,
  }

  return context
}

export default function TanstackQueryProvider({
  children,
  context,
}: {
  children: ReactNode
  context: ReturnType<typeof getContext>
}) {
  const { queryClient } = context

  return (
    <TRPCProvider trpcClient={trpcClient} queryClient={queryClient}>
      {children}
    </TRPCProvider>
  )
}
