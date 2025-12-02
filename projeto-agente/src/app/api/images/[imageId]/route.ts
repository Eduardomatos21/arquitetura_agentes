import { promises as fs } from "fs";
import { NextRequest } from "next/server";
import path from "path";

const DEFAULT_DATASET_PATH = path.join("assets", "STAD_TRAIN_MSIMUT", "MSIMUT");

function resolveImageDir(): string {
  const configured = process.env.TCGA_IMAGE_DIR;
  if (configured && configured.trim().length > 0) {
    return path.isAbsolute(configured)
      ? configured
      : path.join(process.cwd(), configured);
  }
  return path.join(process.cwd(), DEFAULT_DATASET_PATH);
}

const IMAGE_DIR = resolveImageDir();
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
  { params }: { params: Promise<{ imageId: string }> }
): Promise<Response> {
  const { imageId } = await params;
  const safeId = sanitizeImageId(imageId);

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
