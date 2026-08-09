import { NextRequest, NextResponse } from "next/server";
import { SESSION_COOKIE, verifySessionToken } from "@/lib/auth";

// Protects the /admin/* page routes (not /admin/login itself, and not the
// /api/admin/* API routes, which each guard themselves via requireAdmin()
// since middleware can't easily short-circuit a route handler's own logic
// the same way and API routes need JSON 401s, not redirects).
export async function proxy(req: NextRequest) {
  if (req.nextUrl.pathname === "/admin/login") return NextResponse.next();

  const token = req.cookies.get(SESSION_COOKIE.name)?.value;
  const authenticated = await verifySessionToken(token);
  if (!authenticated) {
    const loginUrl = new URL("/admin/login", req.url);
    return NextResponse.redirect(loginUrl);
  }
  return NextResponse.next();
}

export const config = {
  matcher: ["/admin/:path*"],
};
