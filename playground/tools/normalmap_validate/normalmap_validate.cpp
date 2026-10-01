// Normal-map validation for the NCS playground (raylib 6.0, OpenGL 4.3+).
//
// Renders world-space normals of (a) the high-res source mesh and (b) the simplified mesh, with and without the
// baked tangent-space normal map, from several views into an RGBA32F target, then measures the per-pixel angular
// error against (a). Also checks the result survives deformation, using the two tangent-frame update paths the
// engine needs: Jacobian transport (what skinning does) and "positions only" re-derivation (what a neural deformer
// that only outputs positions forces us to do).
//
// Inputs come from playground/tools/blender/make_normalmap_testcase.py (hi.pgm, lo.pgm, normal.png).
// Hardware GPU required; --allow-software runs an API/math-only check on a software rasterizer.
// Deformation is applied on the CPU here for test simplicity; in the engine the same math runs in compute.
#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#define PG_GLAPI __stdcall
#else
#define PG_GLAPI
#endif
typedef unsigned int GLenum; typedef unsigned int GLuint; typedef int GLint; typedef int GLsizei;
#define GL_TRIANGLES 0x0004
#define GL_UNSIGNED_INT 0x1405
#define GL_TEXTURE_2D 0x0DE1
#define GL_RGBA 0x1908
#define GL_FLOAT 0x1406
#define GL_RENDERER 0x1F01
#define GL_VENDOR 0x1F00
#define GL_VERSION 0x1F02
static void (PG_GLAPI *pglDrawElements)(GLenum, GLsizei, GLenum, const void*);
static void (PG_GLAPI *pglBindTexture)(GLenum, GLuint);
static void (PG_GLAPI *pglGetTexImage)(GLenum, GLint, GLenum, GLenum, void*);
static const unsigned char* (PG_GLAPI *pglGetString)(GLenum);
template <class F> static bool load(F& f, const char* n) { f = (F)rlGetProcAddress(n); return f != nullptr; }

// ------------------------------------------------------------------------------------------------ mesh I/O
struct Vtx { float p[3], n[3], uv[2], t[4]; uint32_t weld; };
static_assert(sizeof(Vtx) == 52, "PGM1 vertex record is 13 x 4 bytes");
struct MeshData { std::vector<Vtx> v; std::vector<uint32_t> idx; };
static bool LoadPgm(const std::string& path, MeshData& m) {
    FILE* f = fopen(path.c_str(), "rb"); if (!f) return false;
    char magic[4]; uint32_t nv = 0, ni = 0;
    bool ok = fread(magic, 1, 4, f) == 4 && memcmp(magic, "PGM1", 4) == 0 && fread(&nv, 4, 1, f) == 1 && fread(&ni, 4, 1, f) == 1;
    if (ok) { m.v.resize(nv); m.idx.resize(ni); ok = fread(m.v.data(), sizeof(Vtx), nv, f) == nv && fread(m.idx.data(), 4, ni, f) == ni; }
    fclose(f); return ok;
}

// ------------------------------------------------------------------------------------------------ deformation
// Smooth analytic deformation: twist about the tube axis (Y), bend in the XY plane, then a rigid motion.
// Jacobian by central differences. Max strain ~ tube radius (0.22 m) x max(twist, curvature).
struct DeformParams { float twist = 0, curvature = 0; Vector3 rigidAxis{1, 0, 0}; float rigidAngle = 0; Vector3 translate{0, 0, 0}; };
static DeformParams g_def;
static Vector3 Deform(Vector3 p) {
    const DeformParams& d = g_def;
    const float y0 = 0.45f;
    float a = d.twist * (y0 - p.y), c = cosf(a), s = sinf(a);
    Vector3 q = { c * p.x + s * p.z, p.y, -s * p.x + c * p.z };
    if (d.curvature > 0) {
        float th = d.curvature * (y0 - q.y), R = 1.0f / d.curvature;
        q = { R - (R - q.x) * cosf(th), y0 - (R - q.x) * sinf(th), q.z };
    }
    return Vector3Add(Vector3RotateByAxisAngle(q, Vector3Normalize(d.rigidAxis), d.rigidAngle), d.translate);
}
static void Jacobian(Vector3 p, float J[3][3]) {
    const float e = 1e-4f;
    for (int c = 0; c < 3; c++) {
        Vector3 a = p, b = p; ((float*)&a)[c] += e; ((float*)&b)[c] -= e;
        Vector3 fa = Deform(a), fb = Deform(b);
        J[0][c] = (fa.x - fb.x) / (2 * e); J[1][c] = (fa.y - fb.y) / (2 * e); J[2][c] = (fa.z - fb.z) / (2 * e);
    }
}
static Vector3 MulJ(const float J[3][3], Vector3 v) { return { J[0][0]*v.x + J[0][1]*v.y + J[0][2]*v.z, J[1][0]*v.x + J[1][1]*v.y + J[1][2]*v.z, J[2][0]*v.x + J[2][1]*v.y + J[2][2]*v.z }; }
static Vector3 MulCofJ(const float J[3][3], Vector3 n) {      // cof(J) * n  ∝  J^{-T} n  (normal transform)
    Vector3 c0 = Vector3CrossProduct({J[0][1], J[1][1], J[2][1]}, {J[0][2], J[1][2], J[2][2]});
    Vector3 c1 = Vector3CrossProduct({J[0][2], J[1][2], J[2][2]}, {J[0][0], J[1][0], J[2][0]});
    Vector3 c2 = Vector3CrossProduct({J[0][0], J[1][0], J[2][0]}, {J[0][1], J[1][1], J[2][1]});
    return Vector3Add(Vector3Add(Vector3Scale(c0, n.x), Vector3Scale(c1, n.y)), Vector3Scale(c2, n.z));
}
static Vector3 V(const float* f) { return { f[0], f[1], f[2] }; }
static void Set(float* f, Vector3 v) { f[0] = v.x; f[1] = v.y; f[2] = v.z; }
static Vector3 Orthonormal(Vector3 t, Vector3 n) { return Vector3Normalize(Vector3Subtract(t, Vector3Scale(n, Vector3DotProduct(n, t)))); }

enum class TangentUpdate { Jacobian, PositionsOnly };
static MeshData DeformMesh(const MeshData& in, bool isLow, TangentUpdate mode) {
    MeshData out = in;
    if (!isLow || mode == TangentUpdate::Jacobian) {
        for (auto& v : out.v) {
            float J[3][3]; Jacobian(V(v.p), J);
            Set(v.p, Deform(V(v.p)));
            Vector3 n = Vector3Normalize(MulCofJ(J, V(v.n)));
            Set(v.n, n);
            if (isLow) Set(v.t, Orthonormal(MulJ(J, V(v.t)), n));   // bitangent sign v.t[3] unchanged (det J > 0)
        }
        return out;
    }
    // PositionsOnly: a deformer gives new positions only. Re-derive smooth normals from the deformed geometry
    // (corner-angle-weighted over welded vertices, the same weighting Blender used for the normals it baked
    // against; area weighting gives a measurably different normal), then transport the rest MikkTSpace tangent with the minimal rotation
    // that takes the rest normal to the new normal, and re-orthonormalize.
    uint32_t nw = 0; for (auto& v : in.v) nw = std::max(nw, v.weld + 1);
    std::vector<Vector3> wn(nw, {0, 0, 0});
    for (auto& v : out.v) Set(v.p, Deform(V(v.p)));
    for (size_t i = 0; i + 2 < out.idx.size(); i += 3) {
        const Vtx &a = out.v[out.idx[i]], &b = out.v[out.idx[i + 1]], &c = out.v[out.idx[i + 2]];
        Vector3 pa = V(a.p), pb = V(b.p), pc = V(c.p);
        Vector3 fn = Vector3Normalize(Vector3CrossProduct(Vector3Subtract(pb, pa), Vector3Subtract(pc, pa)));
        const Vtx* cs[3] = {&a, &b, &c}; Vector3 ps[3] = {pa, pb, pc};
        for (int k = 0; k < 3; k++) {
            Vector3 e1 = Vector3Normalize(Vector3Subtract(ps[(k + 1) % 3], ps[k])), e2 = Vector3Normalize(Vector3Subtract(ps[(k + 2) % 3], ps[k]));
            float ang = acosf(std::clamp(Vector3DotProduct(e1, e2), -1.0f, 1.0f));
            wn[cs[k]->weld] = Vector3Add(wn[cs[k]->weld], Vector3Scale(fn, ang));
        }
    }
    for (size_t i = 0; i < out.v.size(); i++) {
        Vector3 n0 = Vector3Normalize(V(in.v[i].n)), n1 = Vector3Normalize(wn[out.v[i].weld]);
        Quaternion q = QuaternionFromVector3ToVector3(n0, n1);
        Set(out.v[i].n, n1);
        Set(out.v[i].t, Orthonormal(Vector3RotateByQuaternion(V(in.v[i].t), q), n1));
    }
    return out;
}

// ------------------------------------------------------------------------------------------------ GPU resources
struct GpuMesh { unsigned int vao = 0, vbo = 0, ebo = 0; int count = 0; };
static GpuMesh Upload(const MeshData& m) {
    GpuMesh g; g.count = (int)m.idx.size();
    g.vao = rlLoadVertexArray(); rlEnableVertexArray(g.vao);
    g.vbo = rlLoadVertexBuffer(m.v.data(), (int)(m.v.size() * sizeof(Vtx)), false);
    rlSetVertexAttribute(0, 3, RL_FLOAT, false, sizeof(Vtx), 0);  rlEnableVertexAttribute(0);
    rlSetVertexAttribute(1, 3, RL_FLOAT, false, sizeof(Vtx), 12); rlEnableVertexAttribute(1);
    rlSetVertexAttribute(2, 2, RL_FLOAT, false, sizeof(Vtx), 24); rlEnableVertexAttribute(2);
    rlSetVertexAttribute(3, 4, RL_FLOAT, false, sizeof(Vtx), 32); rlEnableVertexAttribute(3);
    g.ebo = rlLoadVertexBufferElement(m.idx.data(), (int)(m.idx.size() * 4), false);  // 32-bit indices (rlgl's draw is 16-bit only)
    rlDisableVertexArray();
    return g;
}
static void Unload(GpuMesh& g) { rlUnloadVertexArray(g.vao); rlUnloadVertexBuffer(g.vbo); rlUnloadVertexBuffer(g.ebo); g = {}; }

static const char* kVS = R"(#version 430
layout(location = 0) in vec3 aPos; layout(location = 1) in vec3 aNrm; layout(location = 2) in vec2 aUV; layout(location = 3) in vec4 aTan;
uniform mat4 uMVP;
out vec3 vN; out vec4 vT; out vec2 vUV;
void main() { vN = aNrm; vT = aTan; vUV = aUV; gl_Position = uMVP * vec4(aPos, 1.0); })";
// MikkTSpace-consistent decode: unnormalized interpolated N/T, B = sign * cross(N, T), n = normalize(ts.x*T + ts.y*B + ts.z*N).
static const char* kFS = R"(#version 430
in vec3 vN; in vec4 vT; in vec2 vUV;
uniform sampler2D uNormalMap; uniform int uMode;   // 0 = geometric normal, 1 = normal map, 2 = normal map with green flipped (negative control)
layout(location = 0) out vec4 oNormal;
void main() {
    vec3 n = vN;
    if (uMode != 0) {
        vec3 ts = texture(uNormalMap, vUV).xyz * 2.0 - 1.0;
        if (uMode == 2) ts.y = -ts.y;
        vec3 b = vT.w * cross(vN, vT.xyz);
        n = ts.x * vT.xyz + ts.y * b + ts.z * vN;
    }
    n = normalize(n);
    if (!gl_FrontFacing) n = -n;
    oNormal = vec4(n, 1.0);
})";

struct Target { unsigned int fbo, color, depth; int w, h; };
static Target MakeTarget(int w, int h) {
    Target t{rlLoadFramebuffer(), rlLoadTexture(nullptr, w, h, PIXELFORMAT_UNCOMPRESSED_R32G32B32A32, 1), rlLoadTextureDepth(w, h, false), w, h};
    rlFramebufferAttach(t.fbo, t.color, RL_ATTACHMENT_COLOR_CHANNEL0, RL_ATTACHMENT_TEXTURE2D, 0);
    rlFramebufferAttach(t.fbo, t.depth, RL_ATTACHMENT_DEPTH, RL_ATTACHMENT_TEXTURE2D, 0);
    return t;
}

struct View { const char* name; Vector3 eye, target; };
static std::vector<float> RenderNormals(const Target& t, unsigned int prog, const GpuMesh& m, const View& v, int mode, unsigned int nmTex) {
    rlEnableFramebuffer(t.fbo); rlViewport(0, 0, t.w, t.h); rlActiveDrawBuffers(1);
    rlDisableColorBlend(); rlEnableDepthTest(); rlDisableBackfaceCulling();
    rlClearColor(0, 0, 0, 0); rlClearScreenBuffers();
    Matrix view = MatrixLookAt(v.eye, v.target, {0, 1, 0});
    Matrix proj = MatrixPerspective(35.0 * DEG2RAD, (double)t.w / t.h, 0.05, 10.0);
    rlEnableShader(prog);
    rlSetUniformMatrix(rlGetLocationUniform(prog, "uMVP"), MatrixMultiply(view, proj));
    rlSetUniform(rlGetLocationUniform(prog, "uMode"), &mode, RL_SHADER_UNIFORM_INT, 1);
    int unit = 0; rlSetUniform(rlGetLocationUniform(prog, "uNormalMap"), &unit, RL_SHADER_UNIFORM_SAMPLER2D, 1);
    rlActiveTextureSlot(0); rlEnableTexture(nmTex);
    rlEnableVertexArray(m.vao);
    pglDrawElements(GL_TRIANGLES, m.count, GL_UNSIGNED_INT, nullptr);
    rlDisableVertexArray(); rlDisableShader(); rlDisableFramebuffer();
    rlEnableColorBlend();
    std::vector<float> px((size_t)t.w * t.h * 4);
    pglBindTexture(GL_TEXTURE_2D, t.color); pglGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, px.data());
    return px;
}

// ------------------------------------------------------------------------------------------------ metrics + images
struct Stats { double mean = 0, median = 0, p95 = 0, over10 = 0; size_t n = 0; };
static std::vector<uint8_t> InteriorMask(const std::vector<float>& a, const std::vector<float>& b, int w, int h, int r) {
    std::vector<uint8_t> m((size_t)w * h, 0);
    auto cov = [&](int x, int y) { size_t i = ((size_t)y * w + x) * 4 + 3; return a[i] > 0.5f && b[i] > 0.5f; };
    for (int y = r; y < h - r; y++) for (int x = r; x < w - r; x++) {
        bool ok = true;
        for (int dy = -r; dy <= r && ok; dy++) for (int dx = -r; dx <= r && ok; dx++) ok = cov(x + dx, y + dy);
        m[(size_t)y * w + x] = ok;
    }
    return m;
}
static Stats AngleStats(const std::vector<float>& ref, const std::vector<float>& img, const std::vector<uint8_t>& mask, std::vector<float>* errOut) {
    std::vector<float> e; e.reserve(mask.size());
    if (errOut) errOut->assign(mask.size(), -1.0f);
    for (size_t i = 0; i < mask.size(); i++) if (mask[i]) {
        const float* a = &ref[i * 4]; const float* b = &img[i * 4];
        float d = std::clamp(a[0]*b[0] + a[1]*b[1] + a[2]*b[2], -1.0f, 1.0f);
        float deg = acosf(d) * RAD2DEG; e.push_back(deg); if (errOut) (*errOut)[i] = deg;
    }
    Stats s; s.n = e.size(); if (e.empty()) return s;
    for (float x : e) { s.mean += x; s.over10 += x > 10.0f; }
    s.mean /= e.size(); s.over10 = 100.0 * s.over10 / e.size();
    std::sort(e.begin(), e.end()); s.median = e[e.size() / 2]; s.p95 = e[(size_t)(0.95 * (e.size() - 1))];
    return s;
}
static void WriteLit(const std::vector<float>& n, int w, int h, Image& sheet, int ox, int oy) {   // simple N.L preview
    Vector3 L = Vector3Normalize({0.5f, 0.6f, 0.8f});
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
        const float* p = &n[((size_t)(h - 1 - y) * w + x) * 4];
        unsigned char c = p[3] > 0.5f ? (unsigned char)(255 * (0.12f + 0.88f * std::max(0.0f, p[0]*L.x + p[1]*L.y + p[2]*L.z))) : 40;
        ImageDrawPixel(&sheet, ox + x, oy + y, {c, c, c, 255});
    }
}
static void WriteErr(const std::vector<float>& err, int w, int h, Image& sheet, int ox, int oy) {  // 0 deg black -> 20+ deg red/yellow
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
        float e = err[(size_t)(h - 1 - y) * w + x];
        Color c = e < 0 ? Color{40, 40, 40, 255} : Color{(unsigned char)std::min(255.0f, e / 10.0f * 255), (unsigned char)std::min(255.0f, std::max(0.0f, (e - 10.0f) / 10.0f * 255)), 0, 255};
        ImageDrawPixel(&sheet, ox + x, oy + y, c);
    }
}

static bool IsSoftwareRenderer(const char* r) {
    std::string s = r ? r : ""; for (auto& c : s) c = (char)tolower((unsigned char)c);
    for (const char* k : {"llvmpipe", "softpipe", "swrast", "lavapipe", "swiftshader", "microsoft basic render", "gdi generic", "warp"})
        if (s.find(k) != std::string::npos) return true;
    return false;
}

int main(int argc, char** argv) {
    std::string dir = ".", out = "normalmap_report";
    bool allowSoftware = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--allow-software") allowSoftware = true;
        else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else dir = a;
    }
    SetConfigFlags(FLAG_WINDOW_HIDDEN); SetTraceLogLevel(LOG_WARNING);
    InitWindow(64, 64, "normalmap_validate");
    if (!load(pglDrawElements, "glDrawElements") || !load(pglBindTexture, "glBindTexture") || !load(pglGetTexImage, "glGetTexImage") || !load(pglGetString, "glGetString")) { printf("GL load failed\n"); return 2; }
    const char* renderer = (const char*)pglGetString(GL_RENDERER);
    printf("GPU: %s / %s | GL %s\n", pglGetString(GL_VENDOR), renderer, pglGetString(GL_VERSION));
    if (IsSoftwareRenderer(renderer)) {
        if (!allowSoftware) { printf("FAIL: software rasterizer; run on the target GPU (or --allow-software for an API/math-only check)\n"); return 1; }
        printf("WARNING: SOFTWARE RASTERIZER (--allow-software): API/math-only run, NOT a GPU validation\n");
    }

    MeshData hi, lo;
    if (!LoadPgm(dir + "/hi.pgm", hi) || !LoadPgm(dir + "/lo.pgm", lo)) { printf("cannot load %s/{hi,lo}.pgm\n", dir.c_str()); return 2; }
    Image nmImg = LoadImage((dir + "/normal.png").c_str());
    if (!nmImg.data) { printf("cannot load normal.png\n"); return 2; }
    ImageFormat(&nmImg, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);                 // linear data: never sRGB-decode a normal map
    Texture2D nm = LoadTextureFromImage(nmImg);
    GenTextureMipmaps(&nm); SetTextureFilter(nm, TEXTURE_FILTER_TRILINEAR); SetTextureWrap(nm, TEXTURE_WRAP_CLAMP);
    printf("hi: %zu verts %zu tris | lo: %zu corner-verts %zu tris | normal map %dx%d\n", hi.v.size(), hi.idx.size() / 3, lo.v.size(), lo.idx.size() / 3, nmImg.width, nmImg.height);

    unsigned int prog = rlLoadShaderProgram(kVS, kFS);
    if (!prog) { printf("shader compile failed\n"); return 2; }
    const int W = 768, Hh = 768;
    Target tgt = MakeTarget(W, Hh);
    if (!rlFramebufferComplete(tgt.fbo)) { printf("FBO incomplete\n"); return 2; }
    // Cameras looking at the tube (engine coords: waist at y = 0.45, hem at y = 0); the two seams are at azimuth 45 and 225 deg.
    std::vector<View> views = {
        {"front", {0.0f, 0.25f, 1.2f}, {0, 0.22f, 0}},
        {"seam45", {0.75f, 0.30f, -0.75f}, {0, 0.22f, 0}},   // faces the seam at azimuth 45 deg (Blender +X+Y -> engine +X-Z)
        {"hem_close", {0.0f, 0.02f, 0.55f}, {0, 0.05f, 0.1f}},
    };

    // Deformations: "cloth" = rigid motion + mild bend/twist (max strain ~5-7%, cloth is nearly inextensible);
    // "large_strain" = up to ~45% stretch/shear, reported only: tangent-space maps cannot rescale detail slopes under strain.
    DeformParams identity, cloth, large;
    cloth.twist = 0.2f; cloth.curvature = 1.0f / 3.0f; cloth.rigidAxis = {0.3f, 1.0f, 0.2f}; cloth.rigidAngle = 0.6f; cloth.translate = {0.02f, 0.01f, -0.03f};
    large.twist = 1.2f; large.curvature = 2.2f;
    struct Case { const char* name; DeformParams def; bool deformed; TangentUpdate upd; bool scored; };
    std::vector<Case> cases = {
        {"rest", identity, false, TangentUpdate::Jacobian, true},
        {"rest_positions_only", identity, true, TangentUpdate::PositionsOnly, true},
        {"cloth_jacobian", cloth, true, TangentUpdate::Jacobian, true},
        {"cloth_positions_only", cloth, true, TangentUpdate::PositionsOnly, true},
        {"large_strain_jacobian", large, true, TangentUpdate::Jacobian, false},
        {"large_strain_positions_only", large, true, TangentUpdate::PositionsOnly, false},
    };
    const char* modeName[3] = {"lo_geometric", "lo_normalmap", "lo_normalmap_flipG"};

    FILE* csv = fopen((out + ".csv").c_str(), "w");
    fprintf(csv, "case,view,variant,pixels,mean_deg,median_deg,p95_deg,pct_over_10deg\n");
    int fails = 0;
    std::vector<double> meanGeo(cases.size(), 0), meanNm(cases.size(), 0), meanFlip(cases.size(), 0);
    for (size_t ci = 0; ci < cases.size(); ci++) {
        const Case& c = cases[ci];
        g_def = c.def;
        MeshData h2 = c.deformed ? DeformMesh(hi, false, c.upd) : hi;
        MeshData l2 = c.deformed ? DeformMesh(lo, true, c.upd) : lo;
        GpuMesh gh = Upload(h2), gl = Upload(l2);
        Image sheet = GenImageColor(W * 4, Hh * (int)views.size(), BLACK);
        for (size_t vi = 0; vi < views.size(); vi++) {
            auto ref = RenderNormals(tgt, prog, gh, views[vi], 0, nm.id);
            std::vector<float> img[3]; std::vector<float> err[3];
            for (int m = 0; m < 3; m++) img[m] = RenderNormals(tgt, prog, gl, views[vi], m, nm.id);
            auto mask = InteriorMask(ref, img[0], W, Hh, 2);
            for (int m = 0; m < 3; m++) {
                Stats s = AngleStats(ref, img[m], mask, &err[m]);
                fprintf(csv, "%s,%s,%s,%zu,%.3f,%.3f,%.3f,%.2f\n", c.name, views[vi].name, modeName[m], s.n, s.mean, s.median, s.p95, s.over10);
                printf("  %-20s %-10s %-20s px %7zu | mean %6.2f | median %6.2f | p95 %6.2f | >10deg %5.1f%%\n", c.name, views[vi].name, modeName[m], s.n, s.mean, s.median, s.p95, s.over10);
                (m == 0 ? meanGeo : m == 1 ? meanNm : meanFlip)[ci] += s.mean / views.size();
            }
            // contact sheet row per view: high-res | low geometric | low + normal map | error of low + normal map (0..20 deg)
            int oy = (int)vi * Hh;
            WriteLit(ref, W, Hh, sheet, 0, oy); WriteLit(img[0], W, Hh, sheet, W, oy); WriteLit(img[1], W, Hh, sheet, 2 * W, oy); WriteErr(err[1], W, Hh, sheet, 3 * W, oy);
        }
        ExportImage(sheet, (out + "_" + c.name + ".png").c_str());
        UnloadImage(sheet); Unload(gh); Unload(gl);
    }
    fclose(csv);

    printf("\nmean angular error over views (deg):\n  %-28s %10s %10s %10s %s\n", "case", "geometric", "normalmap", "flipG", "");
    for (size_t ci = 0; ci < cases.size(); ci++)
        printf("  %-28s %10.2f %10.2f %10.2f %s\n", cases[ci].name, meanGeo[ci], meanNm[ci], meanFlip[ci], cases[ci].scored ? "" : "(report only)");
    auto check = [&](const char* what, bool ok) { printf("[%s] %s\n", ok ? "PASS" : "FAIL", what); fails += !ok; };
    const double rest = meanNm[0];
    check("N1 normal map recovers detail: mean error <= 50% of the simplified mesh without it", rest <= 0.5 * meanGeo[0]);
    check("N2 convention: flipped green channel >= 1.5x worse (catches Y+/Y- and bitangent-sign bugs)", meanFlip[0] >= 1.5 * rest);
    check("N3 engine normal recomputation matches the baker (identity deformation, positions-only path) within 1.1x of rest", meanNm[1] <= 1.1 * rest);
    check("N4 cloth motion, skinning-style tangent update (Jacobian): normal map still <= 50% of geometric error", meanNm[2] <= 0.5 * meanGeo[2]);
    check("N5 cloth motion, deformer-style update (positions only): normal map still <= 50% of geometric error", meanNm[3] <= 0.5 * meanGeo[3]);
    printf("info: normal-map error relative to rest: cloth_jacobian %.2fx, cloth_positions_only %.2fx, large_strain_jacobian %.2fx, large_strain_positions_only %.2fx\n",
           meanNm[2] / rest, meanNm[3] / rest, meanNm[4] / rest, meanNm[5] / rest);
    printf("%d failure(s); report: %s.csv, %s_<case>.png\n", fails, out.c_str(), out.c_str());
    CloseWindow();
    return fails;
}
