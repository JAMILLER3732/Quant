import { cookies } from "next/headers";
import { NextResponse } from "next/server";
import { SESSION_COOKIE, verifySessionToken } from "@/lib/auth";

/** Guard for API route handlers: returns a 401 NextResponse if not authenticated, else null. */
export async function requireAdmin(): Promise<NextResponse | null> {
  const store = await cookies();
  const token = store.get(SESSION_COOKIE.name)?.value;
  const ok = await verifySessionToken(token);
  if (!ok) return NextResponse.json({ error: "Not authenticated." }, { status: 401 });
  return null;
}
