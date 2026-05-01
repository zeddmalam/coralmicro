import { createTRPCContext } from '@trpc/tanstack-react-query'
import type { AppRouter } from '@monitoring/api/routers/index'

export const { TRPCProvider, useTRPC } = createTRPCContext<AppRouter>()
