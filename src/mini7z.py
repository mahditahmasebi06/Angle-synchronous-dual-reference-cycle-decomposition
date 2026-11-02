"""Minimal 7z extractor (plain header, single-coder LZMA/LZMA2 folders, no encryption),
written for this project with the Python standard library only. Follows 7zFormat.txt."""
import struct, lzma, sys, os, zlib


class R:
    def __init__(s, b): s.b, s.i = b, 0
    def byte(s): v = s.b[s.i]; s.i += 1; return v
    def bytes(s, n): v = s.b[s.i:s.i + n]; s.i += n; return v
    def num(s):
        first = s.byte(); mask = 0x80; val = 0
        for k in range(8):
            if first & mask == 0:
                hi = first & (mask - 1)
                return val | (hi << (8 * k))
            val |= s.byte() << (8 * k); mask >>= 1
        return val
    def bits(s, n):
        out, cur, m = [], 0, 0
        for _ in range(n):
            if m == 0: cur = s.byte(); m = 0x80
            out.append(bool(cur & m)); m >>= 1
        return out
    def defined(s, n):
        all_def = s.byte()
        return [True] * n if all_def else s.bits(n)


def parse_streams(r):
    info = {}
    while True:
        pid = r.byte()
        if pid == 0x00: return info
        if pid == 0x06:                       # PackInfo
            info['pack_pos'] = r.num(); n = r.num(); sizes = []
            while True:
                q = r.byte()
                if q == 0: break
                if q == 0x09: sizes = [r.num() for _ in range(n)]
                elif q == 0x0A: d = r.defined(n); [r.bytes(4) for x in d if x]
            info['pack_sizes'] = sizes
        elif pid == 0x07:                     # UnPackInfo
            assert r.byte() == 0x0B; nf = r.num(); assert r.byte() == 0
            folders = []
            for _ in range(nf):
                ncoders = r.num(); coders = []
                for _ in range(ncoders):
                    fl = r.byte(); cid = r.bytes(fl & 0x0F)
                    assert not fl & 0x10, 'complex coders unsupported'
                    props = r.bytes(r.num()) if fl & 0x20 else b''
                    coders.append((cid, props))
                folders.append({'coders': coders})
            assert r.byte() == 0x0C
            for f in folders: f['unpack'] = [r.num() for _ in f['coders']]
            q = r.byte()
            if q == 0x0A:
                d = r.defined(nf); [r.bytes(4) for x in d if x]; q = r.byte()
            assert q == 0
            info['folders'] = folders
        elif pid == 0x08:                     # SubStreamsInfo
            nf = len(info['folders']); nums = [1] * nf; q = r.byte()
            if q == 0x0D: nums = [r.num() for _ in range(nf)]; q = r.byte()
            sizes = []
            for f, n in zip(info['folders'], nums):
                if n == 0: continue
                tot = f['unpack'][-1]; part = []
                if q == 0x09:
                    part = [r.num() for _ in range(n - 1)]
                part.append(tot - sum(part)); sizes += part
            if q == 0x09: q = r.byte()
            if q == 0x0A:
                k = sum(nums); d = r.defined(k); [r.bytes(4) for x in d if x]; q = r.byte()
            assert q == 0
            info['sub_sizes'] = sizes; info['sub_nums'] = nums
        else:
            raise ValueError(f'unexpected property {pid:#x}')


def parse_files(r):
    n = r.num(); names = None; empty = [False] * n; emptyfile = None
    while True:
        pid = r.byte()
        if pid == 0: break
        size = r.num(); blk = R(r.bytes(size))
        if pid == 0x0E: empty = blk.bits(n)
        elif pid == 0x0F: emptyfile = blk.bits(sum(empty))
        elif pid == 0x11:
            assert blk.byte() == 0
            raw = blk.b[1:].decode('utf-16-le'); names = raw.split('\x00')[:n]
    return names, empty


def decoder(coder):
    cid, props = coder
    if cid == b'\x03\x01\x01':
        lc_lp_pb = props[0]; dsz = struct.unpack('<I', props[1:5])[0]
        lc = lc_lp_pb % 9; lp = (lc_lp_pb // 9) % 5; pb = lc_lp_pb // 45
        return lzma.LZMADecompressor(lzma.FORMAT_RAW, filters=[{'id': lzma.FILTER_LZMA1, 'dict_size': dsz, 'lc': lc, 'lp': lp, 'pb': pb}])
    if cid == b'\x21':
        p = props[0]; dsz = (2 | (p & 1)) << (p // 2 + 11)
        return lzma.LZMADecompressor(lzma.FORMAT_RAW, filters=[{'id': lzma.FILTER_LZMA2, 'dict_size': dsz}])
    raise ValueError(f'unsupported coder {cid.hex()}')


def extract(path, outdir):
    data = open(path, 'rb').read()
    assert data[:6] == b"7z\xbc\xaf'\x1c"
    off, size, _ = struct.unpack('<QQI', data[12:32])
    r = R(data[32 + off: 32 + off + size])
    pid = r.byte()
    if pid == 0x17:                           # encoded header -> decode then re-parse
        si = parse_streams(r); f = si['folders'][0]
        start = 32 + si['pack_pos']
        hdr = decoder(f['coders'][0]).decompress(data[start:start + si['pack_sizes'][0]], max_length=f['unpack'][-1])
        r = R(hdr); pid = r.byte()
    assert pid == 0x01
    si = {}; names = empty = None
    while True:
        q = r.byte()
        if q == 0: break
        if q == 0x04: si = parse_streams(r)
        elif q == 0x05: names, empty = parse_files(r)
        elif q == 0x02:
            while r.byte() != 0: pass
    pos = 32 + si['pack_pos']; sizes = iter(si['sub_sizes'])
    file_iter = iter([nm for nm, e in zip(names, empty) if not e])
    os.makedirs(outdir, exist_ok=True); n_out = 0
    for f, psz, nsub in zip(si['folders'], si['pack_sizes'], si.get('sub_nums', [1] * len(si['folders']))):
        assert len(f['coders']) == 1, 'multi-coder folders unsupported'
        raw = decoder(f['coders'][0]).decompress(data[pos:pos + psz], max_length=f['unpack'][-1]); pos += psz
        k = 0
        for _ in range(nsub):
            sz = next(sizes); nm = next(file_iter)
            dst = os.path.normpath(os.path.join(outdir, nm.replace('\\', '/')))
            assert dst.startswith(os.path.abspath(outdir)), 'path traversal'
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            open(dst, 'wb').write(raw[k:k + sz]); k += sz; n_out += 1
    for nm, e in zip(names, empty):            # empty files / directories
        if e:
            dst = os.path.normpath(os.path.join(outdir, nm.replace('\\', '/')))
            if '.' in os.path.basename(nm): open(dst, 'wb').close()
            else: os.makedirs(dst, exist_ok=True)
    return n_out


if __name__ == '__main__':
    print(extract(sys.argv[1], os.path.abspath(sys.argv[2])), 'files extracted')
