// raylib 6.0 (GRAPHICS_API_OPENGL_43) capability validation for the NCS playground.
// Checks the GPU features the spec depends on; prints PASS/FAIL per test. Exit code = #failures.
// The MLP weights are random (seeded) and generated at runtime: this is NOT a trained model.
// Hardware GPU required: software rasterizers (llvmpipe, WARP, SwiftShader, ...) FAIL T1 unless
// --allow-software is passed, in which case the run only checks API usage, not the GPU.
// NOTE: never include <windows.h> (or CUDA/GL system headers) in a TU that includes raylib.h;
// Win32 symbols clash with raylib (CloseWindow, Rectangle, ...). CUDA lives in cuda_interop.cpp.
#include "raylib.h"
#include "rlgl.h"
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// ---- raw GL via rlGetProcAddress (proves we can reach any GL entry point raylib doesn't wrap)
#ifdef _WIN32
#define PG_GLAPI __stdcall
#else
#define PG_GLAPI
#endif
typedef unsigned int GLenum; typedef unsigned int GLuint; typedef int GLint; typedef uint64_t GLuint64;
#define GL_VERSION 0x1F02
#define GL_RENDERER 0x1F01
#define GL_SHADING_LANGUAGE_VERSION 0x8B8C
#define GL_TIME_ELAPSED 0x88BF
#define GL_QUERY_RESULT 0x8866
#define GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS 0x90EB
#define GL_MAX_SHADER_STORAGE_BLOCK_SIZE 0x90DE
#define GL_MAX_COMPUTE_SHARED_MEMORY_SIZE 0x8262
#define GL_SHADER_STORAGE_BARRIER_BIT 0x2000
#define GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT 0x1
#define GL_SHADER_IMAGE_ACCESS_BARRIER_BIT 0x20
#define GL_TEXTURE_FETCH_BARRIER_BIT 0x8
#define GL_ALL_BARRIER_BITS 0xFFFFFFFF
#define GL_TEXTURE_2D 0x0DE1
#define GL_RGBA 0x1908
#define GL_FLOAT 0x1406
#define GL_POINTS 0x0000
#define GL_VENDOR 0x1F00
#define GL_SYNC_GPU_COMMANDS_COMPLETE 0x9117
#define GL_SYNC_FLUSH_COMMANDS_BIT 0x00000001
#define GL_ALREADY_SIGNALED 0x911A
#define GL_CONDITION_SATISFIED 0x911C
#define GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS 0x90DB
typedef struct __GLsync* GLsync;
static const unsigned char* (PG_GLAPI *pglGetString)(GLenum);
static void (PG_GLAPI *pglGetIntegerv)(GLenum, GLint*);
static void (PG_GLAPI *pglGenQueries)(int, GLuint*);
static void (PG_GLAPI *pglBeginQuery)(GLenum, GLuint);
static void (PG_GLAPI *pglEndQuery)(GLenum);
static void (PG_GLAPI *pglGetQueryObjectui64v)(GLuint, GLenum, GLuint64*);
static void (PG_GLAPI *pglMemoryBarrier)(GLenum);
static void (PG_GLAPI *pglBindTexture)(GLenum, GLuint);
static void (PG_GLAPI *pglGetTexImage)(GLenum, GLint, GLenum, GLenum, void*);
static void (PG_GLAPI *pglDrawArrays)(GLenum, GLint, int);
static GLenum (PG_GLAPI *pglGetError)(void);
static GLsync (PG_GLAPI *pglFenceSync)(GLenum, unsigned int);
static GLenum (PG_GLAPI *pglClientWaitSync)(GLsync, unsigned int, GLuint64);
static void (PG_GLAPI *pglDeleteSync)(GLsync);
template <class F> static bool load(F& f, const char* n) { f = (F)rlGetProcAddress(n); return f != nullptr; }

#ifdef PG_WITH_CUDA
bool CudaWriteGlBuffer(unsigned int glBuffer, const void* src, size_t bytes, std::string& info); // cuda_interop.cpp
#endif

static int g_fail = 0;
static void report(const char* name, bool ok, const std::string& info = "") {
    printf("[%s] %s %s\n", ok ? "PASS" : "FAIL", name, info.c_str());
    if (!ok) g_fail++;
}

// ---- Implicit neural model: SIREN-style MLP 3 -> H -> H -> H -> 1, random weights (NOT a trained model).
constexpr int IN = 3, H = 64, NL = 4; // 4 linear layers
struct Mlp { std::vector<float> w; std::vector<int> off; };
static Mlp makeMlp(uint32_t seed) {
    std::mt19937 rng(seed); Mlp m; int dims[NL + 1] = {IN, H, H, H, 1};
    for (int l = 0; l < NL; l++) {
        int fi = dims[l], fo = dims[l + 1];
        float b = (l == 0) ? 1.0f / fi : std::sqrt(6.0f / fi) / 30.0f;  // SIREN init
        std::uniform_real_distribution<float> U(-b, b);
        m.off.push_back((int)m.w.size());
        for (int i = 0; i < fo * fi + fo; i++) m.w.push_back(U(rng));   // W (fo x fi) row-major, then bias
    }
    return m;
}
static float mlpCpu(const Mlp& m, const float* p) {
    int dims[NL + 1] = {IN, H, H, H, 1}; float a[H], b[H]; std::memcpy(a, p, sizeof(float) * IN);
    for (int l = 0; l < NL; l++) {
        int fi = dims[l], fo = dims[l + 1]; const float* W = &m.w[m.off[l]]; const float* B = W + fo * fi;
        for (int o = 0; o < fo; o++) { float s = B[o]; for (int i = 0; i < fi; i++) s += W[o * fi + i] * a[i];
            b[o] = (l < NL - 1) ? std::sin(30.0f * s) : s; }
        std::memcpy(a, b, sizeof(float) * fo);
    }
    return a[0];
}
// GLSL MLP shared by compute + fragment shaders: weights read from an SSBO.
static const char* kMlpGlsl = R"(
layout(std430, binding = 0) readonly buffer Weights { float W[]; };
const int H = 64;
float mlp(vec3 p) {
    float a[H]; float b[H];
    a[0] = p.x; a[1] = p.y; a[2] = p.z;
    int off = 0; int fi = 3;
    for (int l = 0; l < 4; l++) {
        int fo = (l == 3) ? 1 : H;
        for (int o = 0; o < fo; o++) {
            float s = W[off + fo*fi + o];
            for (int i = 0; i < fi; i++) s += W[off + o*fi + i] * a[i];
            b[o] = (l < 3) ? sin(30.0 * s) : s;
        }
        for (int o = 0; o < fo; o++) a[o] = b[o];
        off += fo*fi + fo; fi = fo;
    }
    return a[0];
}
)";

static bool IsSoftwareRenderer(const char* r) {
    std::string s = r ? r : "";
    for (auto& c : s) c = (char)tolower((unsigned char)c);
    for (const char* k : {"llvmpipe", "softpipe", "swrast", "lavapipe", "swiftshader", "microsoft basic render", "gdi generic", "warp"})
        if (s.find(k) != std::string::npos) return true;
    return false;
}

int main(int argc, char** argv) {
    bool allowSoftware = false;
    for (int i = 1; i < argc; i++) if (std::string(argv[i]) == "--allow-software") allowSoftware = true;
    SetConfigFlags(FLAG_WINDOW_HIDDEN);
    SetTraceLogLevel(LOG_WARNING);
    InitWindow(256, 256, "rlval");

    bool glOk = load(pglGetString, "glGetString") && load(pglGetIntegerv, "glGetIntegerv") &&
        load(pglGenQueries, "glGenQueries") && load(pglBeginQuery, "glBeginQuery") && load(pglEndQuery, "glEndQuery") &&
        load(pglGetQueryObjectui64v, "glGetQueryObjectui64v") && load(pglMemoryBarrier, "glMemoryBarrier") &&
        load(pglBindTexture, "glBindTexture") && load(pglGetTexImage, "glGetTexImage") &&
        load(pglDrawArrays, "glDrawArrays") && load(pglGetError, "glGetError") &&
        load(pglFenceSync, "glFenceSync") && load(pglClientWaitSync, "glClientWaitSync") && load(pglDeleteSync, "glDeleteSync");
    report("T0 raw GL entry points via rlGetProcAddress", glOk);
    if (!glOk) return 1;

    GLint inv = 0, ssbo = 0, shm = 0, csBlocks = 0;
    pglGetIntegerv(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS, &csBlocks);
    pglGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &inv);
    pglGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &ssbo);
    pglGetIntegerv(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, &shm);
    char info[512];
    const char* renderer = (const char*)pglGetString(GL_RENDERER);
    bool software = IsSoftwareRenderer(renderer);
    snprintf(info, sizeof info, "| %s / %s | GL %s | GLSL %s | maxInvocations %d | CS SSBO blocks %d | maxSSBO %d MB | shared %d KB",
        pglGetString(GL_VENDOR), renderer, pglGetString(GL_VERSION), pglGetString(GL_SHADING_LANGUAGE_VERSION),
        inv, csBlocks, ssbo >> 20, shm >> 10);
    report("T1 hardware GPU, GL 4.3+ core context with compute", rlGetVersion() == RL_OPENGL_43 && (!software || allowSoftware), info);
    if (software)
        printf("       %s\n", allowSoftware ? "WARNING: SOFTWARE RASTERIZER (--allow-software): API-only run, NOT a GPU validation; timings meaningless"
                                            : "software rasterizer detected: run on the target GPU (or pass --allow-software for an API-only check)");

    // T2: shader compile failure is non-fatal (needed for hot reload: keep last good program)
    unsigned int bad = rlLoadShader("#version 430\nvoid main(){ this is not glsl }", RL_COMPUTE_SHADER);
    report("T2 bad shader returns 0 without crashing (hot-reload fallback)", bad == 0);

    // T3: implicit neural model evaluated in a compute shader, parity vs CPU
    Mlp mlp = makeMlp(1234);
    const int G = 48, N = G * G * G;
    std::vector<float> pts(N * 4);
    for (int i = 0; i < N; i++) {
        int x = i % G, y = (i / G) % G, z = i / (G * G);
        pts[i*4+0] = -1 + 2.0f * x / (G - 1); pts[i*4+1] = -1 + 2.0f * y / (G - 1); pts[i*4+2] = -1 + 2.0f * z / (G - 1); pts[i*4+3] = 0;
    }
    unsigned int wBuf = rlLoadShaderBuffer((unsigned)(mlp.w.size() * 4), mlp.w.data(), RL_DYNAMIC_COPY);
    unsigned int pBuf = rlLoadShaderBuffer((unsigned)(pts.size() * 4), pts.data(), RL_DYNAMIC_COPY);
    unsigned int oBuf = rlLoadShaderBuffer((unsigned)(N * 4), nullptr, RL_DYNAMIC_COPY);
    std::string cs = std::string("#version 430\nlayout(local_size_x = 64) in;\n") + kMlpGlsl + R"(
layout(std430, binding = 1) readonly buffer Pts { vec4 P[]; };
layout(std430, binding = 2) writeonly buffer Out { float O[]; };
uniform int n;
void main() { uint i = gl_GlobalInvocationID.x; if (i >= uint(n)) return; O[i] = mlp(P[i].xyz); }
)";
    unsigned int csId = rlLoadShader(cs.c_str(), RL_COMPUTE_SHADER);
    unsigned int prog = csId ? rlLoadShaderProgramCompute(csId) : 0;
    bool t3 = false; std::string t3info;
    if (prog) {
        GLuint q; pglGenQueries(1, &q);
        rlEnableShader(prog);
        int nLoc = rlGetLocationUniform(prog, "n"); rlSetUniform(nLoc, &N, RL_SHADER_UNIFORM_INT, 1);
        rlBindShaderBuffer(wBuf, 0); rlBindShaderBuffer(pBuf, 1); rlBindShaderBuffer(oBuf, 2);
        pglBeginQuery(GL_TIME_ELAPSED, q);
        rlComputeShaderDispatch((N + 63) / 64, 1, 1);
        pglEndQuery(GL_TIME_ELAPSED);
        pglMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        rlDisableShader();
        GLuint64 ns = 0; pglGetQueryObjectui64v(q, GL_QUERY_RESULT, &ns);
        std::vector<float> out(N); rlReadShaderBuffer(oBuf, out.data(), N * 4, 0);
        double maxErr = 0, maxAbs = 0; bool finite = true;
        for (int i = 0; i < N; i++) {
            float ref = mlpCpu(mlp, &pts[i*4]);
            maxErr = std::fmax(maxErr, std::fabs(ref - out[i])); maxAbs = std::fmax(maxAbs, std::fabs(ref));
            finite &= std::isfinite(out[i]);
        }
        char b[256]; snprintf(b, sizeof b, "| %d pts, MLP 3-64-64-64-1 sine | max|err| %.2e (|f|max %.3f) | GPU timer query %.2f ms (software GL)", N, maxErr, maxAbs, ns / 1e6);
        t3info = b; t3 = finite && maxErr < 1e-3;
    }
    report("T3 implicit MLP in compute shader matches CPU (+ GL timer query)", t3, t3info);

    // T4: sphere-trace an implicit surface whose SDF includes the MLP, in a fragment shader reading the SSBO
    std::string vs = R"(#version 430
in vec3 vertexPosition; in vec2 vertexTexCoord; out vec2 uv; uniform mat4 mvp;
void main(){ uv = vertexTexCoord; gl_Position = mvp*vec4(vertexPosition,1.0); })";
    std::string fs = std::string("#version 430\n") + kMlpGlsl + R"(
out vec4 color;
float sdf(vec3 p){ return length(p) - 0.6 + 0.08*mlp(p); }
vec3 nrm(vec3 p){ vec2 e = vec2(1e-3,0); return normalize(vec3(sdf(p+e.xyy)-sdf(p-e.xyy), sdf(p+e.yxy)-sdf(p-e.yxy), sdf(p+e.yyx)-sdf(p-e.yyx))); }
void main(){
    vec2 uv = gl_FragCoord.xy/256.0;
    vec3 ro = vec3(0,0,2.2), rd = normalize(vec3(uv*2.0-1.0, -1.6));
    float t = 0.0; bool hit = false;
    for (int i = 0; i < 64; i++){ float d = sdf(ro+rd*t); if (d < 1e-3){ hit = true; break; } t += d*0.8; if (t > 5.0) break; }
    if (!hit){ color = vec4(0.1,0.1,0.12,1); return; }
    vec3 n = nrm(ro+rd*t); float l = max(dot(n, normalize(vec3(0.6,0.8,0.5))),0.0);
    color = vec4(vec3(0.15) + vec3(0.85,0.75,0.6)*l, 1);
})";
    Shader sh = LoadShaderFromMemory(vs.c_str(), fs.c_str());
    RenderTexture2D rt = LoadRenderTexture(256, 256);
    BeginTextureMode(rt); ClearBackground(BLACK);
    BeginShaderMode(sh); rlBindShaderBuffer(wBuf, 0);
    DrawTextureRec(Texture2D{rlGetTextureIdDefault(), 1, 1, 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8}, Rectangle{0,0,1,1}, Vector2{0,0}, WHITE); // prime batch
    DrawRectangle(0, 0, 256, 256, WHITE);
    EndShaderMode(); EndTextureMode();
    Image img = LoadImageFromTexture(rt.texture); ImageFlipVertical(&img);
    ExportImage(img, "implicit_mlp.png");
    Color* px = LoadImageColors(img); int lit = 0;
    for (int i = 0; i < 256 * 256; i++) if (px[i].r > 60) lit++;
    UnloadImageColors(px);
    char b4[128]; snprintf(b4, sizeof b4, "| shader id %u | %.1f%% surface pixels -> implicit_mlp.png", sh.id, 100.0 * lit / (256 * 256));
    report("T4 sphere-traced neural implicit (fragment shader + SSBO)", sh.id != rlGetShaderIdDefault() && lit > 2000 && lit < 60000, b4);

    // T5: compute shader writes vertex positions into an SSBO that is drawn directly as a vertex buffer (zero copy)
    const int V = 64 * 64;
    unsigned int vBuf = rlLoadShaderBuffer(V * 16, nullptr, RL_DYNAMIC_COPY);
    unsigned int dcs = rlLoadShader(R"(#version 430
layout(local_size_x = 64) in;
layout(std430, binding = 3) buffer Verts { vec4 v[]; };
uniform float time;
void main(){ uint i = gl_GlobalInvocationID.x; float x = float(i % 64u)/63.0*2.0-1.0, y = float(i / 64u)/63.0*2.0-1.0;
  v[i] = vec4(x*0.9, y*0.9 + 0.1*sin(6.0*x + time), 0.0, 1.0); })", RL_COMPUTE_SHADER);
    unsigned int dprog = rlLoadShaderProgramCompute(dcs);
    rlEnableShader(dprog); float tm = 1.0f; rlSetUniform(rlGetLocationUniform(dprog, "time"), &tm, RL_SHADER_UNIFORM_FLOAT, 1);
    rlBindShaderBuffer(vBuf, 3); rlComputeShaderDispatch(V / 64, 1, 1); rlDisableShader();
    pglMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
    unsigned int vao = rlLoadVertexArray(); rlEnableVertexArray(vao);
    rlEnableVertexBuffer(vBuf); rlSetVertexAttribute(0, 4, RL_FLOAT, false, 16, 0); rlEnableVertexAttribute(0);
    rlDisableVertexArray();
    unsigned int pprog = rlLoadShaderProgram(
        "#version 430\nlayout(location=0) in vec4 p; void main(){ gl_Position = p; gl_PointSize = 2.0; }",
        "#version 430\nout vec4 c; void main(){ c = vec4(1,0,0,1); }");
    RenderTexture2D rt2 = LoadRenderTexture(256, 256);
    BeginTextureMode(rt2); ClearBackground(BLACK); rlDrawRenderBatchActive();
    rlEnableShader(pprog); rlEnableVertexArray(vao); pglDrawArrays(GL_POINTS, 0, V); rlDisableVertexArray(); rlDisableShader();
    EndTextureMode();
    Image img2 = LoadImageFromTexture(rt2.texture); Color* px2 = LoadImageColors(img2); int red = 0;
    for (int i = 0; i < 256 * 256; i++) if (px2[i].r > 128 && px2[i].g < 50) red++;
    UnloadImageColors(px2);
    std::vector<float> vread(8); rlReadShaderBuffer(vBuf, vread.data(), 32, 0);
    char b5[160]; snprintf(b5, sizeof b5, "| %d red pixels; v[0]=(%.3f,%.3f) | GL err 0x%x", red, vread[0], vread[1], pglGetError());
    report("T5 compute-deformed SSBO drawn as vertex buffer (GPU deformer path)", red > 1000, b5);

    // T6: float MRT G-buffer (RGBA16F x3 + RGBA32F + depth), write + read back exact values
    unsigned int fbo = rlLoadFramebuffer();
    unsigned int tex[4];
    for (int i = 0; i < 4; i++) {
        tex[i] = rlLoadTexture(nullptr, 64, 64, i < 3 ? PIXELFORMAT_UNCOMPRESSED_R16G16B16A16 : PIXELFORMAT_UNCOMPRESSED_R32G32B32A32, 1);
        rlFramebufferAttach(fbo, tex[i], RL_ATTACHMENT_COLOR_CHANNEL0 + i, RL_ATTACHMENT_TEXTURE2D, 0);
    }
    unsigned int dtex = rlLoadTextureDepth(64, 64, false);
    rlFramebufferAttach(fbo, dtex, RL_ATTACHMENT_DEPTH, RL_ATTACHMENT_TEXTURE2D, 0);
    bool complete = rlFramebufferComplete(fbo);
    unsigned int mrt = rlLoadShaderProgram(
        "#version 430\nconst vec2 q[3]=vec2[](vec2(-1,-1),vec2(3,-1),vec2(-1,3)); void main(){ gl_Position = vec4(q[gl_VertexID],0,1); }",
        "#version 430\nlayout(location=0) out vec4 a; layout(location=1) out vec4 b; layout(location=2) out vec4 c; layout(location=3) out vec4 d;\n"
        "void main(){ a=vec4(0.25,0.5,-1.0,2.0); b=vec4(-0.5); c=vec4(8.0,16.0,0.125,1.0); d=vec4(1234.5678, -3.14159, 1e-4, 65536.0); }");
    rlEnableFramebuffer(fbo); rlViewport(0, 0, 64, 64); rlActiveDrawBuffers(4);
    rlDisableColorBlend(); // G-buffer pass must not blend (raylib enables alpha blending by default)
    unsigned int emptyVao = rlLoadVertexArray();
    rlEnableShader(mrt); rlEnableVertexArray(emptyVao); pglDrawArrays(0x0004 /*GL_TRIANGLES*/, 0, 3); rlDisableVertexArray(); rlDisableShader();
    rlEnableColorBlend();
    rlDisableFramebuffer();
    float a4[4], d4[4]; std::vector<float> buf(64 * 64 * 4);
    pglBindTexture(GL_TEXTURE_2D, tex[0]); pglGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, buf.data()); std::memcpy(a4, buf.data(), 16);
    pglBindTexture(GL_TEXTURE_2D, tex[3]); pglGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, buf.data()); std::memcpy(d4, buf.data(), 16);
    bool vals = a4[0] == 0.25f && a4[2] == -1.0f && a4[3] == 2.0f && std::fabs(d4[0] - 1234.5678f) < 1e-3f && d4[3] == 65536.0f;
    char b6[200]; snprintf(b6, sizeof b6, "| complete=%d | rt0=(%.3f,%.3f,%.3f,%.3f) rt3=(%.4f,%.5f,%.0e,%.0f)", complete, a4[0], a4[1], a4[2], a4[3], d4[0], d4[1], d4[2], d4[3]);
    report("T6 float MRT G-buffer (4 targets + depth), negative/HDR values preserved", complete && vals, b6);

    // T7: image load/store from compute (e.g. writing neural output into a texture)
    unsigned int itex = rlLoadTexture(nullptr, 32, 32, PIXELFORMAT_UNCOMPRESSED_R32G32B32A32, 1);
    unsigned int ics = rlLoadShader(R"(#version 430
layout(local_size_x = 8, local_size_y = 8) in;
layout(rgba32f, binding = 0) uniform writeonly image2D img;
void main(){ ivec2 p = ivec2(gl_GlobalInvocationID.xy); imageStore(img, p, vec4(p, -float(p.x+p.y), 42.0)); })", RL_COMPUTE_SHADER);
    unsigned int iprog = rlLoadShaderProgramCompute(ics);
    rlEnableShader(iprog); rlBindImageTexture(itex, 0, PIXELFORMAT_UNCOMPRESSED_R32G32B32A32, false);
    rlComputeShaderDispatch(4, 4, 1); rlDisableShader(); pglMemoryBarrier(GL_ALL_BARRIER_BITS);
    std::vector<float> ib(32 * 32 * 4);
    pglBindTexture(GL_TEXTURE_2D, itex); pglGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, ib.data());
    int k = (5 * 32 + 7) * 4; // pixel (7,5)
    bool t7 = ib[k] == 7 && ib[k+1] == 5 && ib[k+2] == -12 && ib[k+3] == 42;
    char b7[96]; snprintf(b7, sizeof b7, "| texel(7,5)=(%.0f,%.0f,%.0f,%.0f)", ib[k], ib[k+1], ib[k+2], ib[k+3]);
    report("T7 compute imageStore into rgba32f texture", t7, b7);

    // T8: newer GLSL (#version 450) accepted in the requested 4.3 context when the driver offers it
    unsigned int v450 = rlLoadShader("#version 450\nlayout(local_size_x=1) in; void main(){}", RL_COMPUTE_SHADER);
    report("T8 '#version 450' shaders compile (driver-dependent; NVIDIA exposes 4.6)", v450 != 0);

    // T10: GPU skinning + sparse morph targets in one compute pass (Geno-sized), parity vs CPU reference
    {
        const int NV = 10329, NB = 75, NM = 3;
        struct Vtx { float p[4], n[4]; uint32_t bi[4]; float bw[4]; };
        std::mt19937 rng(7); std::uniform_real_distribution<float> U(-1, 1);
        std::vector<Vtx> vtx(NV);
        for (auto& v : vtx) {
            for (int k = 0; k < 3; k++) { v.p[k] = U(rng); v.n[k] = U(rng); }
            v.p[3] = 1; v.n[3] = 0;
            float sum = 0; for (int k = 0; k < 4; k++) { v.bi[k] = (uint32_t)(rng() % NB); v.bw[k] = 0.05f + std::fabs(U(rng)); sum += v.bw[k]; }
            for (int k = 0; k < 4; k++) v.bw[k] /= sum;
        }
        std::vector<float> bones(NB * 16);                       // column-major mat4: rotation about a random axis + translation
        for (int b = 0; b < NB; b++) {
            float ax[3] = {U(rng), U(rng), U(rng)}; float l = std::sqrt(ax[0]*ax[0] + ax[1]*ax[1] + ax[2]*ax[2]); for (auto& a : ax) a /= l;
            float an = U(rng) * 3.14159f, c = std::cos(an), sn = std::sin(an), t = 1 - c, x = ax[0], y = ax[1], z = ax[2];
            float R[9] = {t*x*x + c, t*x*y - sn*z, t*x*z + sn*y,  t*x*y + sn*z, t*y*y + c, t*y*z - sn*x,  t*x*z - sn*y, t*y*z + sn*x, t*z*z + c};
            float* M = &bones[b * 16];
            for (int col = 0; col < 3; col++) for (int row = 0; row < 3; row++) M[col * 4 + row] = R[row * 3 + col];
            M[3] = M[7] = M[11] = 0; M[12] = U(rng); M[13] = U(rng); M[14] = U(rng); M[15] = 1;
        }
        std::vector<uint32_t> moff(NV + 1, 0); std::vector<float> mdelta;   // vertex-major sparse morphs: (dx,dy,dz,morphIdx)
        for (int i = 0; i < NV; i++) {
            moff[i] = (uint32_t)(mdelta.size() / 4);
            for (int m = 0; m < NM; m++) if (rng() % 3 == 0) { mdelta.insert(mdelta.end(), {0.1f * U(rng), 0.1f * U(rng), 0.1f * U(rng), (float)m}); }
        }
        moff[NV] = (uint32_t)(mdelta.size() / 4);
        float mw[NM] = {0.7f, -0.3f, 1.0f};
        unsigned int bV = rlLoadShaderBuffer(NV * sizeof(Vtx), vtx.data(), RL_STATIC_DRAW);
        unsigned int bB = rlLoadShaderBuffer((unsigned)(bones.size() * 4), bones.data(), RL_DYNAMIC_DRAW);
        unsigned int bO = rlLoadShaderBuffer((unsigned)(moff.size() * 4), moff.data(), RL_STATIC_DRAW);
        unsigned int bD = rlLoadShaderBuffer((unsigned)(mdelta.size() * 4), mdelta.data(), RL_STATIC_DRAW);
        unsigned int bW = rlLoadShaderBuffer(sizeof mw, mw, RL_DYNAMIC_DRAW);
        unsigned int bOut = rlLoadShaderBuffer(NV * 32, nullptr, RL_DYNAMIC_COPY);      // vec4 pos + vec4 normal, drawable as VBO
        unsigned int scs = rlLoadShader(R"(#version 430
layout(local_size_x = 64) in;
struct Vtx { vec4 p; vec4 n; uvec4 bi; vec4 bw; };
struct Out { vec4 p; vec4 n; };
layout(std430, binding = 0) readonly buffer Verts { Vtx v[]; };
layout(std430, binding = 1) readonly buffer Bones { mat4 skin[]; };
layout(std430, binding = 2) readonly buffer MorphOff { uint moff[]; };
layout(std430, binding = 3) readonly buffer MorphDelta { vec4 md[]; };
layout(std430, binding = 4) readonly buffer MorphW { float mw[]; };
layout(std430, binding = 5) writeonly buffer Result { Out o[]; };
uniform int n;
void main() {
    uint i = gl_GlobalInvocationID.x; if (i >= uint(n)) return;
    vec3 p = v[i].p.xyz;
    for (uint k = moff[i]; k < moff[i + 1u]; k++) p += mw[uint(md[k].w)] * md[k].xyz;
    mat4 M = v[i].bw.x * skin[v[i].bi.x] + v[i].bw.y * skin[v[i].bi.y] + v[i].bw.z * skin[v[i].bi.z] + v[i].bw.w * skin[v[i].bi.w];
    o[i].p = vec4((M * vec4(p, 1.0)).xyz, 1.0);
    o[i].n = vec4(mat3(M) * v[i].n.xyz, 0.0);
})", RL_COMPUTE_SHADER);
        unsigned int sprog = scs ? rlLoadShaderProgramCompute(scs) : 0;
        bool ok = sprog != 0; char b10[200] = "| compute skinning shader failed to compile";
        if (ok) {
            GLuint q; pglGenQueries(1, &q);
            rlEnableShader(sprog); rlSetUniform(rlGetLocationUniform(sprog, "n"), &NV, RL_SHADER_UNIFORM_INT, 1);
            rlBindShaderBuffer(bV, 0); rlBindShaderBuffer(bB, 1); rlBindShaderBuffer(bO, 2); rlBindShaderBuffer(bD, 3); rlBindShaderBuffer(bW, 4); rlBindShaderBuffer(bOut, 5);
            pglBeginQuery(GL_TIME_ELAPSED, q); rlComputeShaderDispatch((NV + 63) / 64, 1, 1); pglEndQuery(GL_TIME_ELAPSED);
            rlDisableShader(); pglMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
            GLuint64 ns = 0; pglGetQueryObjectui64v(q, GL_QUERY_RESULT, &ns);
            std::vector<float> out(NV * 8); rlReadShaderBuffer(bOut, out.data(), NV * 32, 0);
            double err = 0;
            for (int i = 0; i < NV; i++) {
                float p[3] = {vtx[i].p[0], vtx[i].p[1], vtx[i].p[2]};
                for (uint32_t k = moff[i]; k < moff[i + 1]; k++) for (int c = 0; c < 3; c++) p[c] += mw[(int)mdelta[k * 4 + 3]] * mdelta[k * 4 + c];
                float M[16] = {0};
                for (int j = 0; j < 4; j++) for (int e = 0; e < 16; e++) M[e] += vtx[i].bw[j] * bones[vtx[i].bi[j] * 16 + e];
                for (int r = 0; r < 3; r++) {
                    float ref = M[r] * p[0] + M[4 + r] * p[1] + M[8 + r] * p[2] + M[12 + r];
                    float refn = M[r] * vtx[i].n[0] + M[4 + r] * vtx[i].n[1] + M[8 + r] * vtx[i].n[2];
                    err = std::fmax(err, std::fmax(std::fabs(ref - out[i * 8 + r]), std::fabs(refn - out[i * 8 + 4 + r])));
                }
            }
            ok = err < 1e-4;
            snprintf(b10, sizeof b10, "| %d verts, %d bones, %d sparse morph deltas | max|err| %.1e | GPU %.3f ms", NV, NB, (int)(mdelta.size() / 4), err, ns / 1e6);
        }
        report("T10 GPU skinning + sparse morph targets (compute, Geno-sized) matches CPU reference", ok, b10);

        // T11: async GPU->CPU readback with a fence (metrics/recording path; never stall the frame)
        GLsync fence = pglFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        GLenum st = pglClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, 1000000000ull);
        pglDeleteSync(fence);
        report("T11 fence sync for deferred readback", st == GL_ALREADY_SIGNALED || st == GL_CONDITION_SATISFIED);
    }

    // T9: CUDA writes into the GL buffer that T5 draws as vertices (zero-copy CUDA deformer / ORT CUDA EP path)
#ifdef PG_WITH_CUDA
    {
        std::vector<float> host(V * 4), back(V * 4);
        for (int i = 0; i < V * 4; i++) host[i] = 0.5f * (float)i;
        std::string cinfo;
        bool ok = CudaWriteGlBuffer(vBuf, host.data(), host.size() * sizeof(float), cinfo);
        if (ok) { rlReadShaderBuffer(vBuf, back.data(), (unsigned)(back.size() * sizeof(float)), 0); ok = std::memcmp(host.data(), back.data(), host.size() * sizeof(float)) == 0; }
        report("T9 CUDA-GL interop: CUDA writes the GL SSBO used as a vertex buffer", ok, "| " + cinfo);
    }
#else
    printf("[SKIP] T9 CUDA-GL interop (configure with -DPG_WITH_CUDA=ON on an NVIDIA machine)\n");
#endif

    CloseWindow();
    printf("\n%d failure(s)\n", g_fail);
    return g_fail ? 1 : 0;
}
