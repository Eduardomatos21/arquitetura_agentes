import { promises as fs } from "fs";
import { NextRequest } from "next/server";
import path from "path";

const IMAGE_DIR = path.join(process.cwd(), "assets", "ISIC-images");
const ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "webp"];

function sanitizeImageId(imageId: string): string | null {
  const safe = imageId.replace(/[^A-Za-z0-9_\-]/g, "");
  return safe.length > 0 ? safe : null;
}

function getContentType(ext: string): string {
  if (ext === "jpg") return "image/jpeg";
  return `image/${ext}`;
}

export async function GET(
  _request: NextRequest,
  { params }: { params: { imageId: string } }
): Promise<Response> {
  const safeId = sanitizeImageId(params.imageId);

  if (!safeId) {
    return new Response("Invalid image id", { status: 400 });
  }

  for (const ext of ALLOWED_EXTENSIONS) {
    const candidatePath = path.join(IMAGE_DIR, `${safeId}.${ext}`);
    try {
      const file = await fs.readFile(candidatePath);
      return new Response(file, {
        status: 200,
        headers: {
          "Content-Type": getContentType(ext),
          "Cache-Control": "public, max-age=86400",
        },
      });
    } catch (error: unknown) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
        console.error(`Failed to load image ${candidatePath}`, error);
        return new Response("Failed to read image", { status: 500 });
      }
    }
  }

  return new Response("Image not found", { status: 404 });
}
