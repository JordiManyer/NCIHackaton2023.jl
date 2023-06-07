

macro index_to_tuple(idx::Integer, D1::Integer, D2::Integer)
	return esc(:(($idx - 1) % $D1 + 1, ($idx - 1) ÷ $D1 + 1))
end

macro index_to_tuple(idx::Integer, D1::Integer, D2::Integer, D3::Integer)
	return esc(:(($idx - 1) % $D1 + 1, (($idx - 1) ÷ $D1) % $D2 + 1, ($idx - 1) ÷ ($D1 * $D2) + 1))
end
