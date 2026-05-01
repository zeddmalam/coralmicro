import { auth } from "@monitoring/auth";

export async function createContext({ req }: { req: Request }) {
  const session = await auth.api.getSession({
    headers: req.headers,
  });
  return {
    auth: null,
    requestUrl: req.url,
    session,
  };
}

export type Context = Awaited<ReturnType<typeof createContext>>;
