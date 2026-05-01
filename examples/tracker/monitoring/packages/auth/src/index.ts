import { createDb } from "@monitoring/db";
import * as schema from "@monitoring/db/schema/auth";
import { env } from "@monitoring/env/server";
import { polar, checkout, portal } from "@polar-sh/better-auth";
import { betterAuth } from "better-auth";
import { drizzleAdapter } from "better-auth/adapters/drizzle";
import { tanstackStartCookies } from "better-auth/tanstack-start";

import { polarClient } from "./lib/payments";

export function createAuth() {
  const db = createDb();

  return betterAuth({
    database: drizzleAdapter(db, {
      provider: "pg",

      schema: schema,
    }),
    trustedOrigins: [env.CORS_ORIGIN],
    emailAndPassword: {
      enabled: true,
    },
    secret: env.BETTER_AUTH_SECRET,
    baseURL: env.BETTER_AUTH_URL,
    plugins: [
      polar({
        client: polarClient,
        createCustomerOnSignUp: true,
        enableCustomerPortal: true,
        use: [
          checkout({
            products: [
              {
                productId: "your-product-id",
                slug: "pro",
              },
            ],
            successUrl: env.POLAR_SUCCESS_URL,
            authenticatedUsersOnly: true,
          }),
          portal(),
        ],
      }),
      tanstackStartCookies(),
    ],
  });
}

export const auth = createAuth();
