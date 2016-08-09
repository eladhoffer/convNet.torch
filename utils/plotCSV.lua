function parseCSV(filename, plotFields, delim)
  local delim = delim or ' | '
  local f = io.open(filename,'r')
  local title = f:read('*l')
  local fields = title:gsub('%s+',' '):split(delim)
